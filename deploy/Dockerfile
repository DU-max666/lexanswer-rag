# 法衡 API 服务镜像
# 构建（律所内网断网环境：使用所内 PyPI 镜像源）：
#   docker build -f deploy/Dockerfile -t lexanswer-api:0.1.0 .
FROM python:3.11-slim

WORKDIR /app

# 仅安装核心依赖；检索/微调重型依赖在各自专用镜像中
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i ${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

COPY corpus/ corpus/
COPY retrieval/ retrieval/
COPY serving/ serving/
COPY verify/ verify/
COPY eval/ eval/

# 非 root 运行
RUN useradd -m lex && chown -R lex:lex /app
USER lex

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# create_app 工厂：无 fastapi 依赖的裸镜像不会误启动
CMD ["sh", "-c", "uvicorn serving.api:create_app --factory --host 0.0.0.0 --port ${PORT}"]
