# 私有化部署手册（案卷不出内网）

## 0. 前提

- 内网 GPU 主机一台：RTX 4090 24GB（vLLM 推理 + 可选微调）；
- 内网 Docker 主机：API / Milvus / MySQL / Redis（可与 GPU 同机）；
- 所内私有镜像 registry；公网镜像先在有审查的出口机拉取、
  扫描后推入 registry，**生产环境本身不出站**。

## 1. 模型权重带入

1. 在联网审查机上执行 `finetune/merge_adapter.py` 得到合并权重；
2. 权重目录经杀毒与完整性审查（SHA256 登记）后，以加密移动介质
   带入内网，放置于 `finetune/output/lexanswer-merged/`；
3. 权重是系统唯一的外部制品，其余全部源码可审计。

## 2. 配置

```bash
cp ../.env.example .env
# 必填：MYSQL_ROOT_PASSWORD / MYSQL_PASSWORD / MINIO_SECRET_KEY
# 全部改为所内密钥管理下发的真实值，禁止使用示例占位符
```

## 3. 启动顺序

```bash
docker compose up -d etcd minio milvus          # 向量库先就绪
docker compose up -d mysql redis                # 业务存储
docker compose up -d vllm                       # 等待 /v1/models 可访问
docker compose up -d api                        # 最后拉起应用层
```

健康检查：

```bash
curl http://127.0.0.1:8080/healthz              # api
curl http://127.0.0.1:8000/v1/models            # vllm
```

## 4. 语料入库

```bash
# 内网机器上运行（Milvus/模型地址走环境变量，配置齐全时不会降级 mock）
LEXANSWER_MILVUS_URI=http://127.0.0.1:19530 \
LEXANSWER_BGE_M3_PATH=/models/bge-m3 \
python -c "
from corpus import load_directory, split_document
from corpus.dedup import dedup_chunks
from retrieval.embeddings import get_embedder
from retrieval.store import get_store
emb = get_embedder(); store = get_store(dim=emb.dim)
chunks = dedup_chunks([c for d in load_directory('corpus/sample_data') for c in split_document(d)]).kept
v = emb.encode([c.text for c in chunks]); store.upsert(chunks, v.dense, v.sparse)
print('入库', len(chunks), '块')
"
```

扫描件目录需接入 OCR 引擎（`rapidocr-onnxruntime`）后重新入库，
`loaders.py` 对未 OCR 文档会打 `needs_ocr` 标记，不会静默丢内容。

## 5. 「案卷不出内网」检查清单

- [ ] vllm / api 容器无公网路由（`docker network inspect` 验证）；
- [ ] `LEXANSWER_CLOUD_BASE_URL` 留空 → 路由层无云端通道；
- [ ] API 端口仅绑定内网入口机，经所内反代 + 内网 SSO 鉴权；
- [ ] 审计日志（`logs/audit.jsonl`）接入所内 SIEM，保留 ≥1 年；
- [ ] MySQL/Redis/Milvus 卷备份介质按保密资产登记，同库存放；
- [ ] 角色表（`LEXANSWER_TOKEN_ROLES`）与 HR 账号系统同步，
      离职当日吊销 token。

## 6. 容量参考（80 律师规模）

| 组件 | 配置 | 说明 |
|---|---|---|
| vLLM | 单卡 4090，max-num-seqs 8 | 峰值并发 4-8 路问答 |
| Milvus | 16GB 内存 | 百万级法条块 HNSW 足够 |
| API | 2C4G ×2 副本 | FastAPI + SSE |
| MySQL/Redis | 8GB 内存共享 | 会话、审计、缓存 |
