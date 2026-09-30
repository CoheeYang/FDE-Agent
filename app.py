# -*- coding: utf-8 -*-
"""FDE 售前智能体 · Streamlit Web 应用
  诊断模式：智能体主动访谈客户（你），侧栏实时显示五阶段进度，可生成/下载诊断报告
  问答模式：客户提问，RAG 顾问带出处作答（可展开查看引用片段）
启动：streamlit run app.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st

import config as C
from fde_agent.advisor import FDEAdvisor
from fde_agent.diagnose import SLASH_HELP, DiagnoseAgent
from fde_agent.knowledge import format_docs, load_vectorstore
from fde_agent.llm import get_embeddings, get_llm
from fde_agent.schemas import stage_progress

OPENING = ("你好，我是 FDE 售前诊断顾问。今天不聊技术，就想搞清楚三件事："
           "钱从哪里漏、人在哪里耗、数据躺在哪里。可以先介绍一下你们是做什么行业的，"
           "以及你自己在里面负责哪块吗？")

st.set_page_config(page_title="FDE 售前智能体", page_icon="🎯", layout="wide")


# ── 资源加载（进程级单例） ────────────────────────────────────────────
@st.cache_resource(show_spinner="加载本地向量模型与知识库…")
def load_core():
    cfg = C.Config()
    embeddings = get_embeddings(cfg)
    store = load_vectorstore(cfg, embeddings)
    return cfg, store


@st.cache_resource(show_spinner=False)
def _llm_cache(temperature: float):
    cfg, _ = load_core()
    return get_llm(cfg, temperature=temperature)


def llm_for(temperature: float):
    try:
        return _llm_cache(temperature)
    except Exception as e:
        st.error(f"LLM 初始化失败：{e}\n\n请检查 .env 中的 FDE_LLM_* 配置。")
        st.stop()


try:
    cfg, store = load_core()
except Exception as e:
    st.error(f"知识库加载失败：{e}\n\n请先运行 `python cli.py ingest` 构建向量库。")
    st.stop()


# ── 会话状态 ─────────────────────────────────────────────────────────
def get_agent() -> DiagnoseAgent:
    if "agent" not in st.session_state:
        agent = DiagnoseAgent(llm_for(0.5), store)
        agent.history.append(("顾问", OPENING))
        st.session_state["agent"] = agent
    return st.session_state["agent"]


def reset_agent():
    st.session_state.pop("agent", None)
    st.session_state.pop("report", None)


# ── 侧栏 ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 FDE 售前智能体")
    st.caption("LangChain + RAG · 知识库：FDE 实战课 / 全景研究报告 / 课程 SOP")
    mode = st.radio("选择模式", ["🩺 诊断访谈（智能体主动提问）", "💬 方法论问答"],
                    index=0 if st.session_state.get("mode", "d") == "d" else 1,
                    horizontal=False, key="mode_radio")
    mode_key = "d" if mode.startswith("🩺") else "a"
    if st.session_state.get("mode") != mode_key:
        st.session_state["mode"] = mode_key

    st.divider()
    st.caption(f"模型：`{cfg.llm_model}`\n\n"
               f"向量：本地 BGE（{store.store.keys().__len__() if hasattr(store, 'store') else '—'} 分块）")

    with st.expander("⚙️ 配置自检（部署排障）"):
        st.markdown(f"- LLM 配置就绪（缓存的 cfg）：{'✅' if cfg.llm_ready else '❌'}")
        _env_m = os.environ.get("FDE_LLM_MODEL", "")
        _env_k = os.environ.get("FDE_LLM_API_KEY", "")
        st.markdown(f"- env FDE_LLM_MODEL：`{_env_m or '（空）'}`")
        st.markdown(f"- env FDE_LLM_API_KEY："
                    f"`{(_env_k[:6] + '…len=' + str(len(_env_k))) if _env_k else '（空）'}`")
        try:
            st.markdown(f"- st.secrets FDE_LLM_MODEL："
                        f"`{st.secrets.get('FDE_LLM_MODEL') or '（None/空）'}`")
        except Exception as e:
            st.markdown(f"- st.secrets：❌ 读取异常 {type(e).__name__}")
        try:
            _c2 = C.Config()
            st.markdown(f"- 重新实例化 Config："
                        f"{'✅ 就绪（' + _c2.llm_model + '）' if _c2.llm_ready else '❌ 未就绪'}")
        except Exception as e:
            st.markdown(f"- 重新实例化 Config：异常 {type(e).__name__}: {e}")

    if mode_key == "d":
        agent = get_agent()
        st.markdown("##### 访谈进度")
        st.code(stage_progress(agent.info), language=None)
        col1, col2 = st.columns(2)
        if col1.button("📄 生成诊断报告", type="primary", use_container_width=True):
            with st.spinner("正在生成报告…"):
                try:
                    st.session_state["report"] = agent.generate_report()
                except Exception as e:
                    st.error(f"报告生成失败：{e}")
        if col2.button("🔄 重新访谈", use_container_width=True):
            reset_agent()
            st.rerun()
    else:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.pop("ask_messages", None)
            st.rerun()

    st.divider()
    st.caption("会话内命令：`/status` 进度 · `/report` 报告\n\n"
               "诊断模式对应方法论：立项四问 → 客户四问 → 真问题挖掘 → 北极星三条件 → 场景三维评分")


# ── 主区：诊断模式 ───────────────────────────────────────────────────
def render_diagnose():
    agent = get_agent()
    st.subheader("🩺 诊断访谈")
    st.caption("智能体将按 FDE 方法论主动采访你（站在客户视角回答即可）。"
               + SLASH_HELP)

    chat = st.container(height=520)
    with chat:
        for who, text in agent.history:
            with st.chat_message("assistant" if who == "顾问" else "user"):
                st.markdown(text)

    if u := st.chat_input("以客户身份回答…"):
        with chat:
            with st.chat_message("user"):
                st.markdown(u)
        with st.spinner("顾问思考中…"):
            try:
                reply = agent.turn(u)
            except Exception as e:
                reply = None
                st.error(f"调用出错：{e}")
        if reply:
            agent.history.append(("客户", u))
            agent.history.append(("顾问", reply))
            st.rerun()

    report = st.session_state.get("report")
    if report:
        st.divider()
        col1, col2 = st.columns([4, 1])
        col1.subheader("📄 售前诊断报告")
        col2.download_button("下载 Markdown", report,
                             file_name="售前诊断报告.md", mime="text/markdown",
                             use_container_width=True)
        with st.expander("点开查看 / 折叠", expanded=True):
            st.markdown(report)


# ── 主区：问答模式 ───────────────────────────────────────────────────
def render_ask():
    st.subheader("💬 方法论问答")
    st.caption("基于 FDE 知识库回答具体问题；要方案前先给指标与基线，顾问才会给落地建议。")

    if "ask_messages" not in st.session_state:
        st.session_state["ask_messages"] = []

    chat = st.container(height=560)
    with chat:
        if not st.session_state["ask_messages"]:
            st.info("试试问：\n\n"
                    "· 我们想做设备预测性维护，第一步该做什么？\n\n"
                    "· 客户只有 50 万预算想做全厂 AI，我该怎么谈？\n\n"
                    "· 怎么给 AI 项目定义验收标准？")
        for role, content, docs in st.session_state["ask_messages"]:
            with st.chat_message(role):
                st.markdown(content)
                if role == "assistant" and docs:
                    with st.expander(f"📚 引用来源（{len(docs)} 条知识片段）"):
                        for i, d in enumerate(docs, 1):
                            m = d.metadata
                            where = f"{m.get('doc', '?')}｜{m.get('section', '')}"
                            if m.get("page"):
                                where += f"｜P{m['page']}"
                            st.markdown(f"**[{i}] {where}**")
                            st.text(d.page_content[:200] + ("…" if len(d.page_content) > 200 else ""))
                            st.divider()

    if q := st.chat_input("输入你的问题…"):
        st.session_state["ask_messages"].append(("user", q, []))
        with st.spinner("检索知识库并思考…"):
            try:
                advisor = FDEAdvisor(llm_for(0.3), store)
                ans, docs = advisor.answer(q, return_docs=True)
            except Exception as e:
                ans, docs = f"[出错] {e}", []
        st.session_state["ask_messages"].append(("assistant", ans, docs))
        st.rerun()


(render_diagnose if st.session_state.get("mode") == "d" else render_ask)()
