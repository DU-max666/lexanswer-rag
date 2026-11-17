"""法衡 · BERT 意图闸门。

位于检索链路最前端的轻量分类器，把用户问题分成四类：

- ``LAW_QUERY``    法条咨询（需要走完整混合检索 + 引用校验）
- ``CASE_SIMILAR`` 类案检索（走类案库 + 语义断崖块）
- ``FAQ``          高频事务问题（优先 BM25 短路，见 faq_shortcircuit）
- ``CHITCHAT``     闲聊/与法律无关（跳过检索，直接轻量回复）

闸门价值：闲聊不进检索链路可省约 70% 的无效向量查询；
FAQ 短路可把高频事务问题的首 Token 延迟压到 0.1s 内。

生产用微调 BERT（4 分类头）；裸环境自动降级为
``HeuristicIntentGate``（规则打分 + 置信度），接口一致。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Intent(str, Enum):
    LAW_QUERY = "law_query"
    CASE_SIMILAR = "case_similar"
    FAQ = "faq"
    CHITCHAT = "chitchat"


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    need_retrieval: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 4),
            "need_retrieval": self.need_retrieval,
        }


class IntentGate:
    """意图闸门接口。"""

    def classify(self, query: str) -> IntentResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 生产路径：微调 BERT
# ---------------------------------------------------------------------------

class BertIntentGate(IntentGate):
    """加载微调后的 4 分类 BERT 检查点。"""

    _LABEL_MAP = {
        "law_query": Intent.LAW_QUERY,
        "case_similar": Intent.CASE_SIMILAR,
        "faq": Intent.FAQ,
        "chitchat": Intent.CHITCHAT,
    }

    def __init__(self, model_path: str, device: Optional[str] = None):
        from transformers import pipeline  # 延迟导入

        self._pipe = pipeline(
            "text-classification", model=model_path,
            device=0 if (device is None and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else -1,
            top_k=1,
        )

    def classify(self, query: str) -> IntentResult:
        out = self._pipe(query[:512])[0]
        intent = self._LABEL_MAP.get(out["label"], Intent.LAW_QUERY)
        return IntentResult(
            intent=intent,
            confidence=float(out["score"]),
            need_retrieval=intent != Intent.CHITCHAT,
        )


# ---------------------------------------------------------------------------
# 降级路径：规则打分
# ---------------------------------------------------------------------------

class HeuristicIntentGate(IntentGate):
    """关键词/正则打分闸门。

    每类意图维护加权特征组，命中累加；归一化为 softmax 式置信度。
    低置信（<0.45）时保守回退 LAW_QUERY——宁可多检索，不可漏检索。
    """

    _FEATURES: Dict[Intent, List] = {
        Intent.LAW_QUERY: [
            (r"第[零一二三四五六七八九十百千0-9]+[条款项]", 3.0),
            (r"《[^》]{2,30}》", 2.5),
            (r"(规定|处罚|法律责任|适用|效力|废止|施行)", 1.5),
            (r"(怎么办|合法|违法|合同|赔偿|侵权)", 1.0),
        ],
        Intent.CASE_SIMILAR: [
            (r"(案例|类案|判决|裁判|判例|起诉|庭审)", 3.0),
            (r"(类似|相近|同样情形|参照)", 2.0),
            (r"(胜诉|败诉|索赔金额)", 1.5),
        ],
        Intent.FAQ: [
            (r"(营业时间|咨询电话|收费标准|怎么委托|委托流程|办公地址|发票|材料清单)", 3.0),
            (r"(收费|费用|报价|预约|多少钱|多久|几天)", 2.0),
            (r"(在哪|哪里|怎么走|几点|地址)", 2.0),
            (r"^(请问|问一下)", 0.5),
        ],
        Intent.CHITCHAT: [
            (r"^(你好|您好|嗨|哈喽|在吗|谢谢|再见|辛苦了)", 3.0),
            (r"(讲个笑话|你是谁|无聊|天气)", 2.5),
            (r"^[^。，！？,.!?]{0,6}$", 1.5),  # 超短句大概率寒暄
        ],
    }

    def __init__(self, min_confidence: float = 0.45):
        self.min_confidence = min_confidence

    def classify(self, query: str) -> IntentResult:
        scores: Dict[Intent, float] = {}
        q = query.strip()
        for intent, feats in self._FEATURES.items():
            s = 0.0
            for pattern, weight in feats:
                if re.search(pattern, q, flags=re.I):
                    s += weight
            scores[intent] = s
        total = sum(scores.values())
        if total <= 0:
            return IntentResult(Intent.LAW_QUERY, 0.3, need_retrieval=True)
        best = max(scores, key=lambda k: scores[k])
        conf = scores[best] / total
        if conf < self.min_confidence:
            best = Intent.LAW_QUERY
            conf = scores[best] / total if scores[best] else self.min_confidence
        return IntentResult(best, conf, need_retrieval=best != Intent.CHITCHAT)


def get_intent_gate(model_path: Optional[str] = None) -> IntentGate:
    """按环境选择闸门：配置了 BERT 检查点则用模型，否则规则降级。"""
    path = model_path or os.environ.get("LEXANSWER_INTENT_BERT_PATH", "")
    if path:
        try:
            return BertIntentGate(path)
        except Exception as exc:
            print(f"[intent_gate] BERT 闸门不可用（{exc}），降级为规则闸门")
    return HeuristicIntentGate()
