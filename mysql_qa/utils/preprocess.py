"""
    jieba分词工具类
        文本预处理的流程
            1.将英文统一转小写  python  Python  PYTHON 关键字匹配时完全不同的词  统一匹配
            2.把句子进行分词

            什么时候调用？
            1.写入redis '分词后问题' 的时候（给bm25算法使用）
            2.用户提交query进行查询的时候
"""
# 导入日志
import logging

# 导入分词库
import jieba


def preprocess_text(text):
    logging.info(f'开始预处理文本 --> {text}')
    try:
        return jieba.lcut(text.lower())
    except Exception as e:
        logging.error(f'文本预处理失败: {e}')
        return []


if __name__ == '__main__':
    text = "AI大模型是什么？"
    print(preprocess_text(text))
