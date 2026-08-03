这里是存放预训练模型和微调后的BERT模型,以及词嵌入模型,
我的模型如下:
bert-base-chinese       --预训练BERT模型(中文文本分类)
bert_query_classifier   --微调后的BERT模型
bge-m3                  --向量化模型, 将问题与文档子块内容转为稠密向量和稀疏向量
bge-reranker-large      --重排序模型

nlp_bert_document-segmentation_chinese-base     --其他推荐模型