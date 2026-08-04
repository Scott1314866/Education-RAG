# 使用官方Python基础镜像
FROM python:3.10.20-slim

# 设置环境变量
# 默认情况：Python 不会立刻打印日志，会攒一堆再输出。防止看不到实时日志
# 不生成缓存文件。例如__pycache__/*.pyc
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 设置工作目录，容器内运行目录，相当于自动跳到工作目录
WORKDIR /app

# 安装系统依赖
# apt update下载的软件包索引缓存，安装完成后直接删掉
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt requirements.txt

# 安装Python依赖
# pip 必须 < 24.1
RUN pip install --no-cache-dir --upgrade "pip < 24.1" && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 清理可能的缓存和临时文件
RUN rm -rf /tmp/* ~/.cache/* /root/.cache/* /opt/miniconda3/ 2>/dev/null || true

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app

# 复制项目文件
COPY . .

# 声明要暴露端口
EXPOSE 8003

# 启动命令，使用环境变量控制主机和端口 -c 告诉 sh "后面跟的是命令字符串，请执行它"
CMD ["sh", "-c", "python app.py"]