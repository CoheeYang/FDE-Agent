# -*- coding: utf-8 -*-
"""诊断访谈智能体：确定性阶段推进（Python 状态机）+ LLM 语言生成与信息抽取。"""
import io
import json
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel as _BaseModel

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore

import config as C
from fde_agent import prompts as P
from fde_agent.knowledge import format_docs
from fde_agent.llm import ask_structured
from fde_agent.schemas import (FIELD_PLAN, DiagnosisInfo, Scenario, STAGES_DONE,
                               next_stage, stage_progress)

SLASH_HELP = ("（会话内命令：/status 查看进度 · /report 生成诊断报告 · /quit 退出）")


class DiagnoseAgent:
    def __init__(self, llm: BaseChatModel, store: Optional[VectorStore] = None,
                 info: Optional[DiagnosisInfo] = None):
        self.llm = llm
        self.store = store
        self.info = info or DiagnosisInfo()
        self.history: List[tuple] = []  # [(who, text)]

    # ── 每轮对话 ──
    def turn(self, user_text: str) -> str:
        user_text = user_text.strip()
        if not user_text:
            return "请说说你们的情况？"
        if user_text == "/status":
            return "访谈进度：\n" + stage_progress(self.info)
        if user_text == "/report":
            return self.generate_report()

        stage = next_stage(self.info)
        stage_hint = ""
        if stage != STAGES_DONE:
            for s, _, _, hint in FIELD_PLAN:
                if s == stage:
                    stage_hint = hint
                    break

        knowledge_block = self._retrieve_block(stage, user_text)
        sys_prompt = P.DIAGNOSE_SYSTEM.format(
            progress_block=stage_progress(self.info),
            next_stage=stage if stage != STAGES_DONE else "（全部完成，准备出报告）",
            stage_hint=stage_hint or "已无待问事项，做收口总结。",
            knowledge_block=knowledge_block or "（本轮无检索摘录）",
        )

        out = ask_structured(self.llm, _TurnOutput, [
            SystemMessage(content=sys_prompt),
            *self._history_messages(),
            HumanMessage(content=P.DIAGNOSE_TURN.format(user_text=user_text)),
        ])
        data = out.model_dump()
        reply = data.pop("reply", "")
        self.info = self._merge(self.info, DiagnosisInfo(**data))
        self.history.append(("客户", user_text))
        self.history.append(("顾问", reply))
        return reply

    def _history_messages(self):
        msgs = []
        for who, text in self.history[-12:]:
            role = HumanMessage if who == "客户" else AIMessage
            msgs.append(role(content=text))
        return msgs

    def _retrieve_block(self, stage: str, user_text: str) -> str:
        if self.store is None:
            return ""
        queries = list(P.STAGE_QUERIES.get(stage, []))
        queries.insert(0, user_text[:80])
        seen, docs = set(), []
        for q in queries:
            try:
                hits = self.store.similarity_search(q, k=3)
            except Exception:
                hits = []
            for d in hits:
                key = d.page_content[:60]
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
        return format_docs(docs[:4])

    @staticmethod
    def _merge(old: DiagnosisInfo, new: DiagnosisInfo) -> DiagnosisInfo:
        data = old.model_dump()
        for k, v in new.model_dump().items():
            if isinstance(v, list):
                if v:
                    data[k] = v
            elif isinstance(v, str):
                if v.strip():
                    data[k] = v.strip()
            elif isinstance(v, dict):
                if any(v.values()):
                    data[k] = v
        return DiagnosisInfo(**data)

    def is_complete(self) -> bool:
        return next_stage(self.info) == STAGES_DONE

    # ── 报告 ──
    def generate_report(self) -> str:
        history_text = "\n".join(f"{w}：{t}" for w, t in self.history) or "（无对话记录）"
        report = ask_structured(self.llm, _ReportOutput, [
            SystemMessage(content=P.REPORT_SYSTEM),
            HumanMessage(content=P.REPORT_USER.format(
                info_json=self.info.model_dump_json(indent=1),
                history_text=history_text[:6000]),
            ),
        ]).markdown
        C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = C.REPORTS_DIR / f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        io.open(str(path), "w", encoding="utf-8").write(report)
        self.report_path = str(path)
        return report + f"\n\n---\n（报告已保存：{path}）"


class _TurnOutput(DiagnosisInfo):
    """每轮结构化输出：更新后的完整状态 + 本轮回复话术。"""
    reply: str = ""


class _ReportOutput(_BaseModel):
    """报告结构化输出。"""
    markdown: str
