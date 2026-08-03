"""
    RAG核心入口

         命令行操作
             进入到cmd ，需要  set PYTHONPATH=E:\work_px\03center_class\edu_rag\edu_rag_pro
         设置项目根目录为E:\work_px\03center_class\edu_rag\edu_rag_pro
        再执行python rag_qa\test.py
"""
import os
from base.config import config
from base.logger import logger
from core.document_processor import process_documents  # 导入处理文档的函数
from core.vector_store import VectorStore
# 老的rag_system模块，该模块测试用它
from core.rag_system import RAGSystem
from openai import OpenAI  # 使用 OpenAI 接口

# RAG核心函数
def main(query_mode=True, directory_path="data"):
    """
        RAG核心函数
        :param query_mode: 查询模式   True：是查询    False：非查询（数据处理）
        :param directory_path: 操作目录
        :return: 无返回值，直接输出回答
    """
    try:
        # 初始化 RAG 系统
        rag_system = RAGSystem()
    except Exception as e:
        logger.error(f"初始化 RAGSystem 失败: {e}")
        print("错误：无法初始化 RAG 系统，无法进入查询模式。")
        return
    # 根据模式执行不同操作
    if not query_mode:
        # --- 数据处理模式 ---
        logger.info("进入数据处理模式...")
        total_chunks_added = 0
        for source_dir in config.VALID_SOURCES:
            dir_path = os.path.join(directory_path, f"{source_dir}_data")
            if os.path.exists(dir_path):
                logger.info(f"开始处理目录: {dir_path}")
                try:
                    chunks = process_documents(
                        dir_path,
                        config.PARENT_CHUNK_SIZE,
                        config.CHILD_CHUNK_SIZE,
                        config.PARENT_CHUNK_OVERLAP,
                        config.CHILD_CHUNK_OVERLAP,
                    )
                    if chunks:
                        rag_system.milvus_client.add_documents(chunks)
                        total_chunks_added += len(chunks)
                        logger.info(f"成功处理目录 {dir_path}，添加了 {len(chunks)} 个文档块")
                    else:
                        logger.info(f"目录 {dir_path} 未发现有效文档或处理结果为空")
                except Exception as e:
                    logger.error(f"处理目录 {dir_path} 时出错: {e}")
            else:
                logger.warning(f"目录 {dir_path} 不存在，跳过处理")
        logger.info(f"数据处理完成，共添加了 {total_chunks_added} 个文档块到向量存储")
    else:
        logger.info("进入交互式查询模式...")
        # 获取有效学科类别
        # valid_sources 有效_学科_类别 ["ai", "java", "test", "ops", "bigdata"]
        valid_sources = config.VALID_SOURCES
        print("\n欢迎使用 EduRAG 交互式查询系统！")
        print(f"支持的学科类别：{valid_sources}")
        print("输入您的问题，或输入 'exit' 退出。")

        while True:
            query = input("\n请输入您的问题：")
            if query.lower() == "exit":
                logger.info("用户退出查询模式")
                print("再见！")
                break
            # ai/java/test/ops/bigdata
            source_filter_input = input(f"请输入学科类别 ({'/'.join(valid_sources)}) (直接回车默认不过滤)：").strip()
            # 学科过滤条件
            source_filter = None  # 默认不过滤
            if source_filter_input:
                if source_filter_input in valid_sources:
                    source_filter = source_filter_input
                    logger.info(f"用户选择了学科过滤: {source_filter}")
                else:
                    logger.warning(
                        f"无效的学科类别 '{source_filter_input}'，将不过滤"
                    )
                    print(f"提示：输入的学科 '{source_filter_input}' 无效，将不过滤。")

            try:
                print("正在生成答案，请稍候...")
                print("-" * 30)
                print(f"问题: {query}")
                print("回答: ")
                # 走 rag_system.py 的 generate_answer 函数
                for chunk in rag_system.generate_answer(query, source_filter):
                    print(chunk, end="")
                print('')
            except Exception as e:
                logger.error(f"处理查询 '{query}' 时失败: {str(e)}")
                print(f"抱歉，处理您的问题时遇到了错误，请稍后重试或联系管理员。\n")


if __name__ == "__main__":
    """
        - 数据处理模式：加载并向量化文档，构建向量数据库，支持多学科目录处理。
        - 查询模式：通过命令行交互式回答用户查询，支持学科过滤。
    """
    # 调用方法1：通过直接调用方式运行
    # 默认进入查询模式
    # 若要执行数据处理，可以修改调用方式，例如：
    # main(query_mode=False, directory_path="./data")
    # 或者通过命令行参数控制

    # 调用方法2：通过脚本传参方式运行
    import argparse
    parser = argparse.ArgumentParser(description="EduRAG System Main Entry Point")
    # store_true：表示存储bool值
    parser.add_argument('--data-processing', action='store_true',
                        help='Run in data processing mode instead of query mode.')
    # 字符串类型，默认值：./data/ai_data
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to the data directory.')
    args = parser.parse_args()
    print('args-->', args)
    main(query_mode=(not args.data_processing), directory_path=args.data_dir)