"""
    文档向量化与存储：core/vector_store.py
"""
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
import os

# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder

from base.config import config
from base.logger import logger


# # 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=config.MILVUS_COLLECTION_NAME,
                 host=config.MILVUS_HOST,
                 port=config.MILVUS_PORT,
                 database=config.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        # rerank_model_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        rerank_model_path = os.path.join(config.MODELS_DIR, 'bge-reranker-large')
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        # device代表设备： mps:m1系列的mac/ cpu: cpu / cuda: nvidia的gpu。和操作系统无关
        self.reranker = CrossEncoder(rerank_model_path, device='cpu')
        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16
        bge_m3_model_path = os.path.join(config.MODELS_DIR, 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=bge_m3_model_path,
            # 在CPU上，FP32往往更稳定、兼容性更好。关闭FP16以保证检索和排序的准确性。
            use_f16=False,
            device='cpu'
        )
        # 获取稠密向量的维度
        self.dense_dim = self.embedding_function.dim["dense"]
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f'http://{self.host}:{self.port}', db_name=self.database)
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    # 定义私有方法，创建或加载 Milvus 集合
    def _create_or_load_collection(self):
        # 检查指定集合是否已存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    def add_documents(self, documents: list[Document]):
        # collection中的每一条数据
        data = []
        # 提取子块内容
        texts = [doc.page_content for doc in documents]  # doc: Document对象
        # 将子块内容 生成 嵌入向量
        embeddings = self.embedding_function(texts)
        for i, doc in enumerate(documents):
            dense_vector = embeddings['dense'][i]
            # 对于稀疏向量, 需要把csr数据转换为 milvus支持的字典类型的数组 {"文档id":系数值}  (这个步骤可以AI)
            sparse_vector = self._convert_sparse_vector(embeddings['sparse'], i)
            # 将数据转化为 Milvus 数据库中 collection集合 可接受的格式
            data.append(
                {
                    "id": hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest(),
                    "text": doc.page_content,
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                    "parent_id": doc.metadata["parent_id"],
                    "parent_content": doc.metadata["parent_content"],
                    "source": doc.metadata["source"],
                    "timestamp": doc.metadata["timestamp"]
                }
            )

            if data:
                self.client.upsert(collection_name=self.collection_name, data=data)
                logger.info(f"已插入 {len(data)} 条数据到集合 {self.collection_name}")
            else:
                logger.error('没有数据插入集合')

    def _convert_sparse_vector(self, sparse_embeddings, index):
        """
        将 BGE-M3 生成的稀疏向量转换为 Milvus 所需的字典格式 {列索引: 值}。
        兼容新版 milvus-model（coo_array）和旧版（csr_matrix）两种格式。

        Args:
            sparse_embeddings: BGE-M3 返回的稀疏向量集合
            index: 当前文档在批次中的索引

        Returns:
            dict: 稀疏向量字典，格式为 {列索引: 非零值}
        """
        sparse_vector = {}
        try:
            # 新版本 milvus-model 使用 coo_array 格式
            row = sparse_embeddings[index]
            # 获取非零元素的列索引数组
            if hasattr(row, 'col'):  # coo_array 格式，新版Milvus
                indices = row.col
            else:  # csr_matrix 格式
                indices = row.indices
        except Exception:
            # 兼容旧版本 milvus-model
            row = sparse_embeddings.getrow(index)
            indices = row.indices
        # 获取稀疏向量的非零值
        values = row.data
        for idx, value in zip(indices, values):
            sparse_vector[int(idx)] = float(value)
        return sparse_vector

    def hybrid_search_with_rerank(self, query, k=config.RETRIEVAL_K, source_filter=None):
        # 对问题分别生成密集向量和稀疏向量
        embeddings = self.embedding_function([query])
        # 稠密向量
        dense_query_vector = embeddings['dense'][0]
        # 稀疏向量
        sparse_query_vector = self._convert_sparse_vector(embeddings['sparse'], 0)

        # 构建查询条件
        # 定义密集向量的查询请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=f'source in ["{source_filter}"]' if source_filter else None,
        )

        # 定义密集向量的查询请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=f'source in ["{source_filter}"]' if source_filter else None,
        )

        ranker = WeightedRanker(1.0, 0.7)

        result = self.client.hybrid_search(
            collection_name=config.MILVUS_COLLECTION_NAME,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["id", "text", "parent_id", "parent_content", "source", "timestamp"],
        )[0]  # result不加[0]的结果是['[{A},{B}...,{N}]']的数据格式, 直接用[0]取出来就行

        # print(result)
        """以上是混合检索的结果"""
        child_chunk = []
        for hit in result:
            child_chunk.append(self._doc_from_hit(hit['entity']))

        parent_docs = self._get_unique_parent_docs(child_chunk)

        if len(parent_docs) < 2:
            return parent_docs[:config.CANDIDATE_M]
        else:
            pairs = [[query, doc.page_content] for doc in parent_docs]
            scores = self.reranker.predict(pairs)
            parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]

            return parent_docs[:config.CANDIDATE_M]

    # 定义私有方法，从 Milvus 查询结果转换成创建 Document 对象
    def _doc_from_hit(self, hit):
        return Document(
            page_content=hit["text"],
            metadata={
                "id": hit["id"],
                "parent_id": hit["parent_id"],
                # 这个是必须要的内容
                "parent_content": hit["parent_content"],
                "source": hit["source"],
                "timestamp": hit["timestamp"],
            }
        )

    # 定义私有方法，从子块中提取去重的父文档，里面只有内容*
    def _get_unique_parent_docs(self, child_chunks):
        parent_contents = set()     # 集合去重
        for chunk in child_chunks:
            parent_contents.add(chunk.metadata.get('parent_content', chunk.page_content))       # metadata是字典，get是方法，用于获取字典中的值
            # 语法格式：dict.get(key, default=None)key：你要在字典中查找的键（Key）。default：可选参数。如果键不存在时，返回这个自定义的默认值；如果不填，默认返回 None
        return [Document(page_content=content) for content in parent_contents]


# 测试
if __name__ == '__main__':
    vector_store = VectorStore()
    # docs = process_documents(config.DATA_DIR + '/ai_data')
    # vector_store.add_documents(docs)

    docs = vector_store.hybrid_search_with_rerank("语⾔模型发展⾛过了哪三个阶段?", 5, "ai")
    print(len(docs))
    print(docs)
