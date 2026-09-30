# -*- coding: utf-8 -*-
"""RAG 问答顾问：检索增强 + 工具调用循环 + 方法论原则约束的回答。"""
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore

from fde_agent import prompts as P
from fde_agent.knowledge import format_docs


def make_search_tool(store: VectorStore, k: int = 4):
    @tool
    def search_fde_knowledge(query: str) -> str:
        """在 FDE 方法论知识库（《FDE 实战课》《FDE 全景研究报告》《FDE 课程 SOP》）中检索。
        输入一个检索词（中文关键词效果最好），返回带编号与出处的知识片段。"""
        hits = store.similarity_search(query, k=k)
        return format_docs(hits) or "（无命中）"
    return search_fde_knowledge


class FDEAdvisor:
    """问答模式：客户问具体问题 → 检索知识库 → 给出带出处的指导。"""

    MAX_TOOL_ROUNDS = 3

    def __init__(self, llm: BaseChatModel, store: VectorStore):
        self.llm = llm
        self.store = store
        self.search_tool = make_search_tool(store)

    def answer(self, question: str, history: Optional[List[tuple]] = None,
               return_docs: bool = False):
        question = question.strip()
        docs = self.store.similarity_search(question, k=5)
        sys_prompt = P.ADVISOR_SYSTEM.format(
            knowledge_block=format_docs(docs) or P.ADVISOR_NO_CONTEXT)
        msgs = [SystemMessage(content=sys_prompt)]
        for who, text in (history or [])[-8:]:
            msgs.append(HumanMessage(content=text) if who == "用户" else AIMessage(content=text))
        msgs.append(HumanMessage(content=question))

        def _finish(text: str):
            return (text, docs) if return_docs else text

        # 工具调用循环：模型可按需追加检索（最多 MAX_TOOL_ROUNDS 轮）
        if isinstance(self.llm, BaseChatModel) and hasattr(self.llm, "bind_tools") \
                and not type(self.llm).__name__.startswith("Scripted"):
            llm_with_tools = self.llm.bind_tools([self.search_tool])
            for _ in range(self.MAX_TOOL_ROUNDS):
                resp = llm_with_tools.invoke(msgs)
                msgs.append(resp)
                calls = getattr(resp, "tool_calls", None) or []
                if not calls:
                    return _finish(resp.content or "")
                for c in calls:
                    result = self.search_tool.invoke(c["args"]["query"]) \
                        if "query" in c["args"] else "（参数缺失）"
                    msgs.append(ToolMessage(content=result, tool_call_id=c["id"]))
            # 超出轮数，强制用已有上下文收口
            msgs.append(HumanMessage(content="请基于以上信息直接给出最终回答，不再检索。"))
            return _finish(self.llm.invoke(msgs).content or "")
        # 脚本化模型（自检）或极简端点：单轮直答
        return _finish(self.llm.invoke(msgs).content or "")
