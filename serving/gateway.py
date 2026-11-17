"""法衡 · 模型网关抽象。

所有生成请求经统一接口 ``ModelGateway`` 下发，不绑定任何厂商：

- ``OpenAICompatGateway``  任何实现了 /v1/chat/completions 的端点
  （vLLM、Ollama、各类云 API 均可），纯标准库 urllib 实现
- ``MockGateway``          离线确定性网关：从上下文中抽取首个法条
  引头，产出符合四段式契约的回答，供单测与评测链路使用

切换只改环境变量（见 .env.example），业务代码零改动。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Dict, Iterator, List, Optional, Union

from corpus.splitters import _int_to_cn, cn_to_int

Message = Dict[str, str]  # {"role": "system|user|assistant", "content": str}


class ModelGateway:
    """网关接口：chat 返回完整文本，stream_chat 返回增量片段迭代器。"""

    name: str = "base"

    def chat(self, messages: List[Message], model: Optional[str] = None,
             temperature: float = 0.2, max_tokens: int = 1024) -> str:
        raise NotImplementedError

    def stream_chat(self, messages: List[Message], model: Optional[str] = None,
                    temperature: float = 0.2, max_tokens: int = 1024) -> Iterator[str]:
        # 默认实现：整段返回后一次性产出（子类可覆写为真流式）
        yield self.chat(messages, model, temperature, max_tokens)


class OpenAICompatGateway(ModelGateway):
    """OpenAI 兼容协议网关（不绑定厂商）。"""

    def __init__(self, base_url: str, api_key: str = "",
                 default_model: str = "lexanswer-7b", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.name = f"openai-compat:{self.base_url}"

    def _request(self, payload: Dict, stream: bool):
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            },
        )
        return urllib.request.urlopen(req, timeout=self.timeout)

    def chat(self, messages, model=None, temperature=0.2, max_tokens=1024) -> str:
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        with self._request(payload, stream=False) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def stream_chat(self, messages, model=None, temperature=0.2, max_tokens=1024):
        """解析 SSE：`data: {...}` 行，`data: [DONE]` 终止。"""
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with self._request(payload, stream=True) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                delta = json.loads(body)["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta


class MockGateway(ModelGateway):
    """离线确定性网关。

    行为可控、可复现：解析 user 消息中的【检索依据】与【问题】，
    - 问题引用的条文在依据中存在 → 引用该条文（微调风格四段式）
    - 不存在 → 微调风格拒答；基座风格则错误引用依据中的首条
      （模拟未微调模型的「该拒不拒、编号漂移」）

    评测 A/B 用 style="base" / "finetuned" 两种回答风格。
    """

    def __init__(self, style: str = "finetuned"):
        self.name = f"mock:{style}"
        self.style = style

    _CITE_RE = re.compile(r"《([^》]{2,30})》第([零一二三四五六七八九十百千0-9]+)条")

    def chat(self, messages, model=None, temperature=0.2, max_tokens=1024) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        head, _, tail = user.partition("【问题】")
        question = tail or user
        ev_cites = [(m.group(1), m.group(2)) for m in self._CITE_RE.finditer(head)]
        q_cites = [(m.group(1), m.group(2)) for m in self._CITE_RE.finditer(question)]
        asked = next(((law, art) for law, art in q_cites if (law, art) in ev_cites), None)
        if asked and self.style == "finetuned":
            law, art = asked
            # 引用依据中的整行条文作为【原文】，保证回查内容重合度达标
            article_line = next(
                (ln.strip() for ln in head.splitlines()
                 if ln.strip().startswith(f"《{law}》第{art}条")),
                f"《{law}》第{art}条（见检索依据）",
            )
            return (f"【原文】{article_line}\n"
                    f"【适用条件】以该条列举的构成要件为准。\n"
                    f"【结论】《{law}》第{art}条现行有效，可作为处理依据。\n"
                    f"【建议】个案适用请承办律师复核，本回复不构成正式法律意见。")
        if self.style == "base":
            # 基座风格：即便依据中有正确条文，编号也常向相邻条文漂移
            src = asked or (ev_cites[0] if ev_cites else None)
            if src:
                law, art = src
                n = cn_to_int(art)
                drift = _int_to_cn(n + 1) if n is not None else art
                return f"关于{law}的规定，见《{law}》第{drift}条，实践中一般可以这样处理，供参考。"
            return f"这个{question[:20]}的问题，通常可以协商或诉讼解决，建议咨询专业人士。"
        # 微调风格：查不到就拒答
        return ("【原文】检索依据中未找到可引用条文。\n"
                "【适用条件】不适用。\n【结论】无法给出有依据的结论。\n"
                "【建议】请核对法名与条号后重试，本回复不构成正式法律意见。")


def get_gateway_from_env() -> ModelGateway:
    """按环境变量装配网关：LEXANSWER_MODEL_BASE_URL → OpenAI 兼容，否则 mock。"""
    base_url = os.environ.get("LEXANSWER_MODEL_BASE_URL", "")
    if base_url:
        return OpenAICompatGateway(
            base_url=base_url,
            api_key=os.environ.get("LEXANSWER_MODEL_API_KEY", ""),
            default_model=os.environ.get("LEXANSWER_MODEL_NAME", "lexanswer-7b"),
        )
    return MockGateway(style=os.environ.get("LEXANSWER_MOCK_STYLE", "finetuned"))
