# -*- coding: utf-8 -*-
"""配置层：LLM/Embedding 提供方、知识源路径、离线模式。
全部可用环境变量覆盖（见 .env.example）。"""
import os
from pathlib import Path

try:  # 读取项目根目录 .env（存在才生效）
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


def _merge_streamlit_secrets() -> None:
    """Streamlit Community Cloud：把 st.secrets 中的配置提升为环境变量（仅填空缺项）。

    st.secrets 为懒加载，Streamlit 要到首次访问时才把 secrets 写入 os.environ；
    若 load_core 的 cache_resource 抢先缓存了未就绪的 Config，会出现
    「环境变量已生效但 cfg 为空」的竞态。本函数在模块导入与 Config 实例化时
    主动触发解析，消除该时序问题。非 Streamlit 环境下静默跳过。
    """
    try:
        import streamlit as _st
        for _k in ("FDE_LLM_BASE_URL", "FDE_LLM_API_KEY", "FDE_LLM_MODEL",
                   "FDE_EMBEDDING_PROVIDER", "FDE_EMBEDDING_BASE_URL",
                   "FDE_EMBEDDING_API_KEY", "FDE_EMBEDDING_MODEL", "HF_ENDPOINT",
                   "ZHIPUAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
            if not os.environ.get(_k):
                _v = _st.secrets.get(_k)
                if _v is not None:
                    os.environ[_k] = str(_v)
    except Exception:
        pass


_merge_streamlit_secrets()

PROJECT_ROOT = Path(__file__).resolve().parent

# HuggingFace 直连在国内网络常超时：在任何 HF 相关库导入前固定走镜像。
# huggingface_hub 在 import 时读取 HF_ENDPOINT，进程内再改无效，故必须在配置层设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
DATA_DIR = Path(os.environ.get("FDE_DATA_DIR", str(PROJECT_ROOT / "data")))
VECTORSTORE_PATH = DATA_DIR / "vectorstore.json"
REPORTS_DIR = DATA_DIR / "reports"

# ── 知识源（默认指向本机的 FDE 资料；缺哪份就跳过哪份，不会中断） ──
# ① FDE 课程 SOP（项目内置副本，对应"经验文档"）
SOP_PATH = Path(os.environ.get(
    "FDE_SOP_PATH",
    str(PROJECT_ROOT / "knowledge" / "FDE课程关键经验与客户洞察-项目全生命周期SOP.md")))
# ② FDE 实战课课程稿（如本机存在则收录）
COURSE_DIR = Path(os.environ.get(
    "FDE_COURSE_DIR", r"C:\Users\Administrator\Desktop\FDE-PPT\md"))
# ③ FDE 全景研究报告：优先用 md 章节源（更干净），不存在再退回 PDF
REPORT_MD_DIR = Path(os.environ.get(
    "FDE_REPORT_MD_DIR", r"C:\Users\Administrator\Desktop\FDE研究报告\md"))
REPORT_PDF = Path(os.environ.get(
    "FDE_REPORT_PDF",
    r"C:\Users\Administrator\Desktop\FDE研究报告\FDE全景研究报告-模式案例与落地操作手册.pdf"))

# ── 离线模式（自检用：确定性哈希向量 + 脚本化模型，不访问网络） ──
def _env_offline() -> bool:
    return os.environ.get("FDE_OFFLINE", "").strip() in ("1", "true", "True")


OFFLINE = _env_offline()  # 供导入期引用；Config 内部运行期重读

# ── 各家 OpenAI 兼容端点的预设 ──
PROVIDER_PRESETS = {
    "bigmodel": {  # 智谱 GLM
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.7",
        "embedding_model": "embedding-3",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "embedding_model": None,  # DeepSeek 无 embedding 接口，需另行配置
    },
    "dashscope": {  # 阿里云百炼（OpenAI 兼容模式）
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "embedding_model": "text-embedding-v3",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-32k",
        "embedding_model": None,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
}

_API_KEY_VARS = {
    "bigmodel": ("ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}


class Config:
    """运行时配置。优先级：FDE_* 显式变量 > 按 API key 自动识别的提供方预设。
    Embedding 提供方：FDE_EMBEDDING_PROVIDER=api（默认，走 OpenAI 兼容接口）
                      或 local（本地 fastembed + BGE 中文小模型，适合 DeepSeek 等无 embedding 的提供方）。"""

    def __init__(self):
        _merge_streamlit_secrets()  # 主动触发 secrets 解析，避免懒加载竞态
        self.llm_base_url = os.environ.get("FDE_LLM_BASE_URL", "")
        self.llm_api_key = os.environ.get("FDE_LLM_API_KEY", "")
        self.llm_model = os.environ.get("FDE_LLM_MODEL", "")
        self.emb_provider = os.environ.get("FDE_EMBEDDING_PROVIDER", "api").strip().lower()
        self.emb_base_url = os.environ.get("FDE_EMBEDDING_BASE_URL", "")
        self.emb_api_key = os.environ.get("FDE_EMBEDDING_API_KEY", "")
        self.emb_model = os.environ.get("FDE_EMBEDDING_MODEL", "")
        self.offline = _env_offline()
        if not self.llm_api_key:
            for name, keys in _API_KEY_VARS.items():
                for k in keys:
                    v = os.environ.get(k)
                    if v:
                        p = PROVIDER_PRESETS[name]
                        self.llm_api_key = v
                        self.llm_base_url = self.llm_base_url or p["base_url"]
                        self.llm_model = self.llm_model or p["model"]
                        if not self.emb_model and p["embedding_model"]:
                            self.emb_model = p["embedding_model"]
                        if not self.emb_api_key:
                            self.emb_api_key = v
                            self.emb_base_url = self.emb_base_url or p["base_url"]
                        break
                if self.llm_api_key:
                    break
        # embedding 未显式配置时，沿用 LLM 的接入点
        if not self.emb_base_url:
            self.emb_base_url = self.llm_base_url
        if not self.emb_api_key:
            self.emb_api_key = self.llm_api_key

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)

    @property
    def embeddings_ready(self) -> bool:
        return bool(self.emb_api_key and self.emb_base_url and self.emb_model)

    def describe(self) -> str:
        if self.offline:
            return "离线模式（哈希向量 + 脚本化模型，仅用于自检）"
        return (f"LLM: {self.llm_model} @ {self.llm_base_url}\n"
                f"Embedding: {self.emb_model or '(未配置)'} @ {self.emb_base_url}")
