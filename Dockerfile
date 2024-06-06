# 使用官方的Python镜像作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制requirements.txt到工作目录
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用程序代码到工作目录
COPY . .

# 暴露FastAPI默认端口
EXPOSE 8000

# 运行FastAPI应用
CMD ["python", "-m", "app.chat_megumi"]
