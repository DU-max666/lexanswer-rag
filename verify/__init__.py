"""法衡 · 引用校验包：抽取、回查、结构校验。"""
from .citation_extractor import Citation, extract_citations
from .citation_checker import (
    CitationCheck, CitationStatus, MODEL_REFUSAL_MARKERS, REFUSAL_ANSWER,
    VerifyVerdict, check_answer, is_model_refusal,
)
from .answer_format import FormatReport, check_answer_format, repair_prompt_for, split_sections

__all__ = [
    "Citation", "extract_citations",
    "CitationCheck", "CitationStatus", "MODEL_REFUSAL_MARKERS", "REFUSAL_ANSWER",
    "VerifyVerdict", "check_answer", "is_model_refusal",
    "FormatReport", "check_answer_format", "repair_prompt_for", "split_sections",
]
