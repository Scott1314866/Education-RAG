"""
    RAG核心系统入口
"""

from openai import OpenAI
from base.config import config
from base.logger import logger
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.query_classifier import QueryClassifier
from rag_qa.core.strategy_selector import StrategySelector
from rag_qa.core.vector_store import VectorStore
import json


class RAGSystem:
    def __init__(self):
        self.query_classifier = QueryClassifier(model_path=f'{config.MODELS_DIR}/bert_query_classifier')
        self.rag_prompt = RAGPrompts.rag_prompt()
        # 初始化大语言模型客户端
        self.llm = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL)
        # 实例化一个策略选择器
        self.strategy_selector = StrategySelector()
        # 初始化Milvus向量数据库
        self.milvus_client = VectorStore(config.MILVUS_COLLECTION_NAME, config.MILVUS_HOST, config.MILVUS_PORT,
                                         config.MILVUS_DATABASE_NAME, )

    def _retrieve_with_hyde(self, query, source_filter=None):
        """
            对问题生成假设答案, 再将生成的假设答案到Milvus数据库中进行检索, 检索出2个文档返回
        """
        try:
            logger.info(f'假设检索数据源: {source_filter}, 原始问题: {query}')
            # 获取假设问题生成提示词模板
            hyde_prompt = RAGPrompts.hyde_prompt().format(query=query)
            hyde_answer = self._call_llm(hyde_prompt)
            logger.info(f'Hyde 生成假设答案: {hyde_answer}')
            """开始通过假设答案到Milvus中进行检索"""
            return self.milvus_client.hybrid_search_with_rerank(
                query=hyde_answer,
                source_filter=source_filter,
            )
        except Exception as e:
            logger.error(f"Hyde 检索异常: {e}")
            return None

    def _retrieve_with_subqueries(self, query, source_filter=None):
        try:
            logger.info(f'子查询检索数据源: {source_filter}, 原始问题: {query}')
            # 获取原始问题生成提示词模板
            subquery_prompt = RAGPrompts.subquery_prompt().format(query=query)
            subquery_answer = self._call_llm(subquery_prompt)
            # logger.info(f'子查询生成问题: {subquery_answer}')
            # 将返回的各个问题存到列表里
            subqueries = [q.strip() for q in subquery_answer.split('\n')]
            if not subqueries:
                return []
            # 定义列表存储所有子问题检索到的文档
            all_docs = []
            for sub_q in subqueries:
                logger.info(f'子查询: {sub_q}')
                """开始到Milvus中进行检索"""
                docs = self.milvus_client.hybrid_search_with_rerank(
                    query=sub_q,
                    source_filter=source_filter,
                )
                # 添加到列表
                all_docs.extend(docs)  # [Document0,.., Document7]
            # 对获取文档进行去重
            unique_docs = {doc.page_content: doc for doc in all_docs}
            # 还原成document
            unique_docs = list(unique_docs.values())
            return unique_docs
        except Exception as e:
            logger.error(f"子查询检索异常: {e}")
            return None

    def _retrieve_with_backtracking(self, query, source_filter=None):
        """
            对问题backtracking简化, 再把简化后的问题 在Milvus数据库中进行检索, 检索出2个文档返回
        """
        try:
            logger.info(f'回溯检索数据源: {source_filter}, 原始问题: {query}')
            # 获取假设问题生成提示词模板
            backtracking_prompt = RAGPrompts.backtracking_prompt().format(query=query)
            backtracking_answer = self._call_llm(backtracking_prompt)
            logger.info(f'backtracking生成简化问题: {backtracking_answer}')
            """开始通过backtracking简化问题再到Milvus中进行检索"""
            return self.milvus_client.hybrid_search_with_rerank(
                query=backtracking_answer,
                source_filter=source_filter,
            )
        except Exception as e:
            logger.error(f"backtracking 检索异常: {e}")
            return None

    def retrieve_and_merge(self, query, strategy, source_filter=None):
        """策略路由函数, 返回最终检索到的文档"""
        logger.info(f"开始处理问题: '{query}'，学科过滤: {source_filter}")
        if strategy == '回溯问题检索':
            rank_sub_chunk = self._retrieve_with_backtracking(query, source_filter)
        elif strategy == '子查询检索':
            rank_sub_chunk = self._retrieve_with_subqueries(query, source_filter)
        elif strategy == '假设问题检索':
            rank_sub_chunk = self._retrieve_with_hyde(query, source_filter)
        else:
            rank_sub_chunk = self.milvus_client.hybrid_search_with_rerank(
                query=query,
                source_filter=source_filter,
            )
        logger.info(f'检索到的文档最终数量为: {len(rank_sub_chunk[:config.CANDIDATE_M])}')
        return rank_sub_chunk[:config.CANDIDATE_M]

    def generate_answer(self, query, source_filter=None, history=None):
        """处理历史
                    大模型存在上下文窗口长度限制，需兼顾对话连贯性与上下文溢出问题，
                    业界主流的方法为
                        滑动窗口法：保留最新n轮对话。缺点是会丢失比较靠前的对话记录
                            摘要压缩法：压缩历史记录，只保留重要的部分。缺点是忘记最新对话细节
                        实际：结合滑动窗口法与摘要压缩法，通过设置窗口容量和摘要长度，来达到既保留重要对话，又能知道最新对话细节
                """
        # 构造历史上下文 把多条历史条目拼接成字符串
        wind_size = 5  # 记忆窗口大小
        # 将历史记录转换成json字符串, ensure_ascii=False 显示中文, 防止中文乱码
        history_context = json.dumps(history, ensure_ascii=False)
        # 如果历史记录长度大于wind_size，则进行摘要压缩
        if history and len(history) > wind_size:
            history_prompt = """
                       "你是一个对话摘要专家。请将以下多轮对话压缩为一段简洁摘要。
                          **保留内容**：
                          - 用户明确询问过的事实信息及其答案（数字、价格、日期、名称、结论）
                          - 用户表现出的偏好、约束条件或身份特征
                          - 尚未解决或用户表示后续会追问的问题
                          **丢弃内容**：
                          - 寒暄、感谢、告别等社交用语
                          - 重复表述或对同一事实的多次确认
                          - 与核心问题无关的闲聊
                          **格式要求**：
                          - 用一段连续文字输出，信息密度优先，不遗漏任何关键事实
                          - 使用"用户询问了…"、"已确认…"、"待跟进…"等客观句式
                          - 不编造对话中不存在的信息

                          要压缩的对话记录为:\n\n
                    """
            history_context = self._call_llm(history_prompt + history_context)
            # 清空历史记录 [压缩成history_context就够了, history可以清空给下一次满滑动窗口大小用]
            history.clear()
            logger.warning(f"压缩对话历史: {history_context[:100]}...")

        logger.info(f"开始处理问题: '{query}'，学科过滤: {source_filter}")
        # 问题进入到bert进行意图识别，识别出通用问题或者专业咨询
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"问题：{query} 问题分类结果: {query_category}")
        # 如果是通用问题，直接调用大模型
        if query_category == "通用知识":
            context = "",
            logger.info("问题为通用问题，调用大模型"),
        else:  # 如果是专业咨询，调用知识库
            strategy = self.strategy_selector.select_strategy(query)
            context_docs = self.retrieve_and_merge(query, strategy, source_filter)
            context = '\n\n'.join([doc.page_content for doc in context_docs])
            logger.info(f"问题为专业咨询，调用知识库，最终检索到的文档数量为: {len(context_docs)}")

        prompt_input = self.rag_prompt.format(
            context=context,
            history=history_context,
            question=query,
            phone=config.CUSTOMER_SERVICE_PHONE
        )
        for text in self._call_llm(prompt_input):
            yield text
        """
        Tips:
            Python 生成器的“惰性执行”.
            只要函数体内出现了 yield，整个函数就是生成器函数.
        """

    def _call_llm(self, prompt, stream=False):
        completion = self.llm.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": prompt}
            ],
            stream=stream,
            extra_body={'enable_think': False}
        )  # completion 是一个生成器
        # for chunk in completion:
        #     # 整个函数是个生成器函数
        #     # 一次 yield 只返回一个
        #     if not chunk.choices:
        #         continue
        #     token = chunk.choices[0].delta.content
        #
        #     yield token
        if stream:
            return (chunk.choices[0].delta.content for chunk in completion if
                    chunk.choices and chunk.choices[0].delta.content)
        else:
            return completion.choices[0].message.content

    # 流式输出 stream=True
    def _call_llm2(self, prompt, stream=True):
        completion = self.llm.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True,
            extra_body={"enable_think": False}
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


if __name__ == '__main__':
    rag_system = RAGSystem()
    # 存储历史的列表
    history = [{"question": "java的课程学费有多少", "answer": "java的课程学费是1000元"},
               {"question": "java的课程如何学习",
                "answer": "java的课程学习方法为：1.学习java基础2.学习java框架3.学习java项目"},
               ]
    # for text in rag_system.generate_answer('Python课程费用多少'):
    #     print(text, end='')

    # print(rag_system._retrieve_with_hyde("ai在教育领域的应用有哪些？"))

    # print(rag_system._retrieve_with_backtracking("100亿的数据可以放到Milvus里么？"))

    # print(rag_system._retrieve_with_subqueries("Java和Python的优缺点？"))
    query = "ai大模型课程大纲?"
    ai_context = ""
    for chunk in rag_system.generate_answer(query, history=history):
        # 拼接ai回复的内容
        ai_context += chunk
        print(chunk, end='')
    # 添加历史记录
    history.append({"question": query, "answer": ai_context})
    pass
