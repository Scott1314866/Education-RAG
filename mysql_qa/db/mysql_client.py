# 导入mysql模块
import pandas as pd
import pymysql

from base.config import config
from base.logger import logger


class MySQLClient:
    def __init__(self):
        try:
            self.connection = pymysql.connect(
                host=config.MYSQL_HOST,
                port=3307,
                user="root",
                password="123456",
                db="subjects_kg"
            )
            # 创建游标对象
            self.cursor = self.connection.cursor()
            # 打印日志
            logger.info("---MySQL---连接成功---")
        except pymysql.MySQLError as e:
            logger.error(f'---MySQL---连接失败---{e}')
            raise

    def creat_table(self):
        create_table_query = '''
                    CREATE TABLE IF NOT EXISTS jpkb (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        subject_name VARCHAR(20),
                        question VARCHAR(1000),
                        answer VARCHAR(1000))
                    '''
        try:
            res = self.cursor.execute(create_table_query)
            logger.info(f"表创建成功:{res}")
        except pymysql.MySQLError as e:
            logger.error(f"表创建失败: {e}")
            raise

    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            for _, row in data.iterrows():
                # logger.info(f"row: {row}")
                sql = 'INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)'
                res = self.cursor.execute(sql, (row['学科名称'], row['问题'], row['答案']))

            # 提交事务
            self.connection.commit()
            logger.info(f"数据插入成功:{res}")
        except Exception as e:
            logger.error(f"数据插入失败: {e}")
            self.connection.rollback()
            raise


    def fetch_questions(self):
        try:
            # 执行SQL
            self.cursor.execute('SELECT question FROM jpkb')
            # 获取结果
            res = self.cursor.fetchall()
            print(type(res))
            print(f'所有问题:{res}')
            # 其中一个元素：('用上下文管理器实现函数运行时间的计算?',)
            # 记录获取成功
            logger.info("成功获取问题")
            # 返回结果
            return res
        except pymysql.MySQLError as e:
            # 记录查询失败
            logger.error(f"查询失败: {e}")
            # 返回空列表
            return []

    def fetch_answer(self, question):
        try:
            # 执行查询
            self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s", (question,))
            # 获取结果，一条记录是一个tuple
            result = self.cursor.fetchone()
            # 返回答案或 None
            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 记录答案获取失败
            logger.error(f"答案获取失败: {e}")
            # 返回 None
            return None

    def close(self):
        try:
            # 关闭连接
            self.connection.close()
            # 记录关闭成功
            logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            logger.error(f"关闭连接失败: {e}")


if __name__ == '__main__':
    mysql_client = MySQLClient()
    # mysql_client.creat_table()
    # mysql_client.insert_data(r'D:\WorkSpace\Education-RAG\mysql_qa\data\JP学科知识问答.csv')
    # mysql_client.fetch_questions()
    answer = mysql_client.fetch_answer('如何在 Ubuntu 中快速创建Pycharm桌面快捷方式?')
    print(f'answer:{answer}')