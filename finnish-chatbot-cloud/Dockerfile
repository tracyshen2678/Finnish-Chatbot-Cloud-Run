FROM python:3.10

# 使用root用户执行
USER root

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量，指定 Hugging Face 缓存目录
ENV HF_HOME=/tmp/transformers_cache

# 设置工作目录
WORKDIR /app

# 配置 Git 用户名和邮箱
RUN git config --global user.email "516904203@qq.com" && \
    git config --global user.name "your_huggingface_username" 

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 安装 uvicorn（如果没有安装）
RUN pip install uvicorn

# 开放端口
EXPOSE 7860

# 启动应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

