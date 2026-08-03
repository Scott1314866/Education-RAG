"""
    意图识别模块: 将问题分类为通用问题和专业咨询问题

    提供如下功能：
    1. 数据加载：读取 5000 条 JSON 数据集，包含查询和标签（“通用知识”或“专业咨询”）
    2. 模型训练：使用 bert-base-chinese 模型，微调二分类任务，准确率达 90%+
    3. 评估优化：直接处理数字标签（0 或 1），生成分类报告和混淆矩阵
    4. 预测接口：支持实时分类，集成到 EduRAG 系统。

    为了满足以上功能，需要实现以下需求：
    1. 初始化方法：初始化预训练的分词器、 预训练的模型。 如果是在上线阶段，主要是负责加载训练好的模型
    2. 数据预处理：将查询文本和预测标签转化为模型的输入数据格式
    3. 构建数据集：用于模型的训练，适配模型的训练函数
    4. 模型训练：基于处理好的数据集划分出来训练集，对模型进行训练
    5. 模型评估：在数据集划分出来的验证集，对模型进行评估
    6. 模型预测：加载训练好的模型，完成意图识别任务
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer

from base.config import config
from base.logger import logger


class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """样本格式: {'input_ids': 'xxxx', 'token_type_ids':'xxxx', 'attention_mask': 'xxxx', 'label':0/1}"""
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class QueryClassifier:
    def __init__(self, model_path='models/bert_query_classifier'):
        self.model_path = model_path
        self.pretrained_model_path = os.path.join(config.MODELS_DIR, 'bert-base-chinese')
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
        self.load_model()
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.label_map = {'通用知识': 0, '专业咨询': 1}
        self.label_map = {'通用知识': 0, '专业咨询': 1}

    def load_model(self):
        # 判断模型路径是否存在
        if os.path.exists(self.model_path):
            # 存在, 直接加载该模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 将模型移动到设备
            self.model.to(self.device)
            logger.info(f'加载模型成功, 模型路径: {self.model_path}')
        else:
            # 不存在, 加载预训练模型
            logger.info("正在初始化模型...")
            # 初始化预训练模型, 设置分类为2, 自动添加输出数量为2的线性层
            self.model = BertForSequenceClassification.from_pretrained(self.pretrained_model_path, num_labels=2)
            self.model.to(self.device)
            logger.info(f"加载预训练模型成功, 预训练模型路径:{self.pretrained_model_path}")

    def train_model(self, data_file='model_generic_5000.json'):
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(value) for value in f.readlines()]

        # 获取问题和标签
        texts = [value['query'] for value in data]
        labels = [value['label'] for value in data]
        # 划分数据集
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2,
                                                                            random_state=42)
        # {'input_ids': all_xxxx, 'token_type_ids':xxxx, 'attention_mask': xxxx}
        # 文本预处理
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
        # 自定义数据集
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)
        # 取出训练数据
        # print(train_dataset[3])

        # 设置训练参数 工程经验参数
        training_args = TrainingArguments(
            # 为了防止模型中途崩溃, 设置检查点
            output_dir='./bert_results',
            # 设置最多保存一次模型, 超出自动删除旧的检查点
            save_total_limit=1,
            # 设置训练轮数
            num_train_epochs=3,
            # 设置训练/验证的批次大小
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            # 设置学习率预热步数(不预热会震荡), 步数表示每一个批次的样本数, 通过预热步数训练初期学习率从0开始, 然后逐渐增加目标值
            # 模型刚开始权重完全随机, 直接给学习率容易损失震荡
            warmup_steps=500,  # 经验: 预热步数=总训练样本的10%
            # 权重惩罚, 防止过拟合
            weight_decay=0.01,
            # 设置日志存放路径
            logging_dir='./bert_logs',
            # 每10步保存一次日志
            logging_steps=10,
            # 评估策略 -> 在每一轮训练完成之后完成一次评估
            eval_strategy="epoch",
            # 保存策略, 为每个epoch结束后保存
            save_strategy="epoch",
            # 设置训练结束后加载最佳模型
            load_best_model_at_end=True,
            # 设置用于判断最佳模型的指标为评估损失
            metric_for_best_model="eval_loss",
            # 禁用FP16混合精度训练，使用FP32精度
            fp16=False,
        )

        # 定义训练器
        trainer = Trainer(
            # 传入要训练的模型
            model=self.model,
            # 传入训练参数
            args=training_args,
            # 传入训练集
            train_dataset=train_dataset,
            # 传入验证集
            eval_dataset=val_dataset,
            # 评估指标
            compute_metrics=self.compute_metrics,
        )

        # 训练模型
        trainer.train()
        # 保存模型
        self.save_model()
        # 评估模型
        self.evaluate_model(val_dataset, val_labels)

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        # 获取预测结果和标签
        logits, labels = eval_pred
        # 获取概率最大的预测结果
        predictions = np.argmax(logits, axis=-1)
        # 计算准确率
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )

        return encodings, [self.label_map[label] for label in labels]

    def create_dataset(self, encodings, labels):
        return Dataset(encodings, labels)

    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f'保存模型成功, 模型路径: {self.model_path}')

    def evaluate_model(self, eval_dataset: Dataset, labels: list[int]):
        """
            评估模型性能: 精确率, 召回率, 准确率, f1分数, ROC曲线
        """
        true_labels = eval_dataset.labels
        trainer = Trainer(model=self.model)
        predict_object = trainer.predict(eval_dataset)
        pred_labels = np.argmax(predict_object.predictions, axis=-1)
        logger.info('\n' + classification_report(true_labels, pred_labels, target_names=['通用知识', '专业咨询']))

    def predict_category(self, query):
        encodings = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt',
        )
        # 将值转换为指定设备
        encodings = {key: value.to(self.device) for key, value in encodings.items()}
        # 不计算梯度, 进行预测
        with torch.no_grad():
            outputs = self.model(**encodings)
            # item() 取值 -> 标量
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            return '通用知识' if prediction == 0 else '专业咨询'


if __name__ == "__main__":
    # 初始化分类器
    classifier = QueryClassifier(model_path="../models/bert_query_classifier")
    # 训练模型
    # classifier.train_model('../classify_data/model_generic_5000.json')
    # 模型预测
    for query in [
        "AI学科的课程大纲是什么",
        "Python课程费用多少？",
        "5*9等于多少？",
        "AI培训有哪些老师？",
    ]:
        classifier.predict_category(query)
        print(f'{query} -> {classifier.predict_category(query)}')
