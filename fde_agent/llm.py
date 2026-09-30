# -*- coding: utf-8 -*-
"""模型工厂：真实模式用 langchain_openai（兼容智谱/DeepSeek/百炼/本地 vLLM 等任意
OpenAI 兼容端点），离线模式提供确定性哈希向量与脚本化模型，供无 Key 自检。"""
from typing import Any, List, Optional

from pydantic import Field

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from config import Config


def get_llm(cfg: Config, temperature: float = 0.4) -> BaseChatModel:
    if cfg.offline:
        raise RuntimeError("离线模式不提供真实 LLM，请在 selftest 中注入 ScriptedChatModel")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        temperature=temperature,
        timeout=90,
        max_retries=2,
    )


def get_embeddings(cfg: Config) -> Embeddings:
    if cfg.offline:
        return OfflineEmbeddings()
    if cfg.emb_provider == "local" or not cfg.embeddings_ready:
        if cfg.emb_provider != "local" and not cfg.embeddings_ready:
            print("[提示] 未配置 API Embedding，使用本地向量模型（fastembed + BGE 中文小模型）。")
        return LocalEmbeddings()
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=cfg.emb_model,
        api_key=cfg.emb_api_key,
        base_url=cfg.emb_base_url,
        timeout=90,
        max_retries=2,
    )


_LOCAL_EMBED_MODEL = None


def LocalEmbeddings() -> Embeddings:
    """本地向量模型（fastembed ONNX，BAAI/bge-small-zh-v1.5，约 100MB，首次自动下载）。
    进程内单例；下载走 HF，失败自动切换 hf-mirror 镜像重试。"""
    global _LOCAL_EMBED_MODEL
    if _LOCAL_EMBED_MODEL is not None:
        return _LOCAL_EMBED_MODEL

    import os
    import config as C
    cache_dir = str(C.DATA_DIR / "models")

    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5", cache_dir=cache_dir)

    import numpy as np

    from langchain_core.embeddings import Embeddings as _E

    class _Local(_E):
        def embed_documents(self, texts):
            vecs = list(model.embed(texts, batch_size=32))
            return [v.tolist() for v in vecs]

        def embed_query(self, text):
            return list(next(iter(model.embed([text]))))

    _LOCAL_EMBED_MODEL = _Local()
    return _LOCAL_EMBED_MODEL


def ask_structured(llm: BaseChatModel, schema: type, messages: List[Any]) -> Any:
    """结构化输出统一入口。
    生产：with_structured_output(method="function_calling") —— 对各家 OpenAI 兼容端点
    （智谱/DeepSeek/百炼）兼容性最好的通道；
    自检：ScriptedChatModel 走脚本化通道。"""
    if isinstance(llm, ScriptedChatModel):
        return llm.pop_structured(schema)
    return llm.with_structured_output(schema, method="function_calling").invoke(messages)


class ScriptedChatModel(BaseChatModel):
    """自检用脚本化模型：按调用顺序弹出预置回复。
    预置项为 ("text", str) 或 ("structured", pydantic 实例)。"""

    script: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fde-scripted"

    def _next(self):
        if not self.script:
            raise RuntimeError("ScriptedChatModel：脚本已耗尽")
        kind, payload = self.script.pop(0)
        return kind, payload

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        kind, payload = self._next()
        content = payload if kind == "text" else str(payload)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def pop_structured(self, schema: type) -> Any:
        kind, payload = self._next()
        if kind != "structured":
            raise RuntimeError("ScriptedChatModel：此处期望 structured 预置")
        if not isinstance(payload, schema):
            raise RuntimeError(f"脚本预置类型 {type(payload).__name__} != 期望 {schema.__name__}")
        return payload

    def bind_tools(self, tools, **kwargs) -> Runnable:  # 自检下禁用工具环
        return RunnableLambda(lambda messages: self.invoke(messages))


class OfflineEmbeddings(Embeddings):
    """确定性哈希向量（signed hashing）。无网络依赖，词元级重叠可产生相似度，
    足够验证 RAG 管线连通性；生产请配置真实 embedding。"""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> List[float]:
        import hashlib
        import math
        import re
        v = [0.0] * self.dim
        for tok in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:16], 16)
            i = h % self.dim
            v[i] += 1.0 if (h >> 8) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)
