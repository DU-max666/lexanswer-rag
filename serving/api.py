"""法衡 · FastAPI 服务层。

串联全链路的对外入口：

    请求 → RBAC 鉴权 → 限流 → 意图闸门/FAQ 短路 → 混合检索
        → 敏感级路由（本地7B/云端） → 生成 → 引用回查/结构校验 → 返回

能力：
- ``POST /v1/chat``        JSON 一次性返回
- ``POST /v1/chat/stream`` SSE 流式返回（末事件附校验结论）
- RBAC：token → 角色 → 可见敏感级（partner/associate/assistant/client）
- 限流：按 token 令牌桶
- 越权审计：敏感级不足或触发拒答的请求写入 JSONL 审计日志

FastAPI 为可选依赖：裸环境导入本模块不报错，仅在实际
``create_app()`` 时给出安装提示，保证 py_compile 与单测链路。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from corpus.schema import Sensitivity
from retrieval.embeddings import get_embedder
from retrieval.faq_shortcircuit import FaqShortCircuit
from retrieval.intent_gate import Intent, get_intent_gate
from retrieval.rerank import get_reranker
from retrieval.store import QueryFilter, get_store
from retrieval.strategies import RetrievalPipeline
from verify.answer_format import check_answer_format
from verify.citation_checker import check_answer, is_model_refusal
from .gateway import Message, ModelGateway, get_gateway_from_env
from .router import route

#: 角色 → 可见的最高敏感级（越级访问写入审计并拒绝）
ROLE_SENSITIVITY: Dict[str, Sensitivity] = {
    "partner": Sensitivity.CASE,      # 合伙人：全部
    "associate": Sensitivity.CASE,    # 执业律师：全部
    "assistant": Sensitivity.INTERNAL,  # 律师助理：公开+内部
    "client": Sensitivity.PUBLIC,     # 客户门户：仅公开法规
}

_SENSITIVITY_RANK = {Sensitivity.PUBLIC: 0, Sensitivity.INTERNAL: 1, Sensitivity.CASE: 2}

SYSTEM_PROMPT = (
    "你是律所内部法律知识助手。仅依据【检索依据】作答，"
    "按【原文】【适用条件】【结论】【建议】四段输出；"
    "检索依据不足时必须拒答，禁止编造法条。"
)


class RateLimiter:
    """按 key 的令牌桶限流（默认 10 req/s，桶容量 20）。"""

    def __init__(self, rate: float = 10.0, capacity: float = 20.0):
        self.rate = rate
        self.capacity = capacity
        self._buckets: Dict[str, List[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self._buckets.get(key, [self.capacity, now])
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens < 1.0:
            self._buckets[key] = [tokens, now]
            return False
        self._buckets[key] = [tokens - 1.0, now]
        return True


class AuditLog:
    """JSONL 审计日志：越权、拒答、校验失败均留痕（不落敏感原文）。"""

    def __init__(self, path: str = "logs/audit.jsonl"):
        self.path = path

    def write(self, event: str, token: str, role: str, detail: str,
              query_len: int = 0) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event, "token": token[:8] + "…",
            "role": role, "detail": detail, "query_len": query_len,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class AuthContext:
    token: str
    role: str


class ChatService:
    """与 Web 框架解耦的核心服务：便于无 fastapi 环境单测。"""

    def __init__(self, pipeline: RetrievalPipeline, gateway: ModelGateway,
                 audit: Optional[AuditLog] = None,
                 tokens: Optional[Dict[str, str]] = None):
        self.pipeline = pipeline
        self.gateway = gateway
        self.audit = audit or AuditLog()
        self.limiter = RateLimiter()
        self.tokens = tokens or self._default_tokens()

    @staticmethod
    def _default_tokens() -> Dict[str, str]:
        """LEXANSWER_TOKEN_ROLES='tok1:partner,tok2:client' 解析。"""
        mapping: Dict[str, str] = {}
        for pair in os.environ.get("LEXANSWER_TOKEN_ROLES", "").split(","):
            if ":" in pair:
                tok, role = pair.split(":", 1)
                mapping[tok.strip()] = role.strip()
        return mapping

    # ---- 鉴权与限流 ----
    def authenticate(self, token: str) -> Optional[AuthContext]:
        role = self.tokens.get(token)
        if role is None or role not in ROLE_SENSITIVITY:
            return None
        return AuthContext(token, role)

    # ---- 主流程 ----
    def ask(self, question: str, auth: AuthContext,
            stream: bool = False):
        if not self.limiter.allow(auth.token):
            self.audit.write("rate_limited", auth.token, auth.role, "触发限流", len(question))
            raise PermissionError("请求过于频繁，请稍后再试")
        max_sens = ROLE_SENSITIVITY[auth.role]
        flt = QueryFilter()
        outcome = self.pipeline.run(question, flt)

        # 闲聊：意图闸门已拦截，不进检索与生成链路，省算力也防幻觉
        if outcome.intent == Intent.CHITCHAT:
            return {"action": "chitchat",
                    "answer": ("您好，我是律所内部法律知识助手，"
                               "可解答法条咨询、类案检索与委托流程问题，"
                               "请描述您的法律问题。"),
                    "citations": [], "format": None}

        # FAQ 短路：人工审定答案直接返回
        if outcome.faq_hit is not None:
            return {"action": "faq", "answer": outcome.faq_hit.entry.answer,
                    "citations": [], "format": None}

        evidences = outcome.evidences
        # 敏感级越权检查：命中块超过角色可见级 → 过滤并审计
        allowed = [ev for ev in evidences
                   if _SENSITIVITY_RANK[ev.parent.sensitivity] <= _SENSITIVITY_RANK[max_sens]]
        if len(allowed) != len(evidences):
            self.audit.write("sensitivity_denied", auth.token, auth.role,
                             f"{len(evidences) - len(allowed)} 个超权限块被过滤", len(question))
        evidences = allowed

        decision = route(
            question,
            retrieved_sensitivities=[ev.parent.sensitivity.value for ev in evidences],
        )
        context = outcome.context()
        messages: List[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【检索依据】\n{context}\n\n【问题】\n{decision.masked_query}"},
        ]
        if stream:
            return self._stream(messages, evidences, decision.target)

        answer = self.gateway.chat(messages)
        return self._finalize(answer, evidences, auth, decision.target)

    def _finalize(self, answer: str, evidences, auth: AuthContext, model_target: str) -> Dict:
        verdict = check_answer(answer, evidences)
        fmt = check_answer_format(answer)
        if is_model_refusal(answer):
            # 模型主动拒答（微调习得能力），原文透传给用户
            return {"action": "refuse", "answer": answer,
                    "citations": [str(c.citation) for c in verdict.checks],
                    "model": model_target, "format": fmt.as_dict()}
        if verdict.action == "refuse":
            self.audit.write("refused", auth.token, auth.role, "引用回查失败触发拒答")
            return {"action": "refuse", "answer": verdict.final_answer,
                    "citations": [str(c.citation) for c in verdict.checks],
                    "model": model_target, "format": fmt.as_dict()}
        if verdict.action == "clarify":
            return {"action": "clarify", "answer": verdict.final_answer,
                    "citations": [str(c.citation) for c in verdict.checks],
                    "model": model_target, "format": fmt.as_dict()}
        return {"action": "answer", "answer": answer,
                "citations": [str(c.citation) for c in verdict.checks],
                "model": model_target, "format": fmt.as_dict()}

    def _stream(self, messages, evidences, model_target: str):
        """SSE 生成器：增量输出正文，末事件附校验结论。"""
        chunks: List[str] = []
        for delta in self.gateway.stream_chat(messages):
            chunks.append(delta)
            yield {"event": "delta", "data": delta}
        full = "".join(chunks)
        result = self._finalize(full, evidences, AuthContext("stream", "partner"), model_target)
        yield {"event": "done", "data": json.dumps(result, ensure_ascii=False)}


def build_default_service() -> ChatService:
    """按环境变量装配默认服务（全 mock 时也可运行）。"""
    embedder = get_embedder()
    store = get_store(dim=getattr(embedder, "dim", 256))
    faq: Optional[FaqShortCircuit] = None
    faq_path = os.environ.get("LEXANSWER_FAQ_PATH", "")
    if faq_path and os.path.exists(faq_path):
        faq = FaqShortCircuit()
        faq.load_jsonl(faq_path)
    pipeline = RetrievalPipeline(embedder, store, get_reranker(), get_intent_gate(), faq)
    return ChatService(pipeline, get_gateway_from_env())


def create_app():
    """FastAPI 应用工厂；未安装 fastapi 时给出提示。"""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("启动 API 需要 `pip install fastapi uvicorn`") from exc

    app = FastAPI(title="法衡 LexAnswer API", version="0.1.0")
    service = build_default_service()

    class ChatRequest(BaseModel):
        question: str
        stream: bool = False

    def require_auth(authorization: str = Header(default="")) -> AuthContext:
        token = authorization.removeprefix("Bearer ").strip()
        auth = service.authenticate(token)
        if auth is None:
            raise HTTPException(status_code=401, detail="无效或缺失的访问令牌")
        return auth

    @app.post("/v1/chat")
    def chat(req: ChatRequest, auth: AuthContext = Depends(require_auth)):
        try:
            return service.ask(req.question, auth)
        except PermissionError as e:
            raise HTTPException(status_code=429, detail=str(e))

    @app.post("/v1/chat/stream")
    def chat_stream(req: ChatRequest, auth: AuthContext = Depends(require_auth)):
        gen = service.ask(req.question, auth, stream=True)

        def sse():
            for ev in gen:
                yield f"event: {ev['event']}\ndata: {ev['data']}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "gateway": service.gateway.name}

    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
