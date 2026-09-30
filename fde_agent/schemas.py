# -*- coding: utf-8 -*-
"""诊断访谈的结构化状态（Pydantic schema）。
字段设计直接对应 FDE 方法论的两组筛子：
  立项四问（报告 7 章 / 课程 05 讲）+ 客户四问（课程 05 讲），
以及真问题五条判断（课程 05 讲）与北极星三条件（课程 06 讲）。"""
from typing import List, Optional

from pydantic import BaseModel, Field


class Scenario(BaseModel):
    name: str = Field("", description="候选场景名，用客户业务语言而非技术语言")
    metric: str = Field("", description="北极星指标（改善什么、用什么数字衡量）")
    baseline: str = Field("", description="基线：指标现在的水平与数据来源")
    value_estimate: str = Field("", description="业务价值估算（可钱化则钱化）")
    data_feasibility: str = Field("", description="数据可行性初判（可得/需治理/缺失）")
    time_to_value: str = Field("", description="见效周期初判")
    score: str = Field("", description="三维评分：价值/可行性/可见性（各1-5，简要）")
    verdict: str = Field("", description="建议：做 / 延后（附重启条件） / 放弃（附理由）")


class DiagnosisInfo(BaseModel):
    # 阶段一：客户画像
    industry: str = Field("", description="客户行业与业务简介")
    interviewee_role: str = Field("", description="受访者角色（决策人/执行层/IT）")
    top_pain: str = Field("", description="客户最想改善的业务指标或最痛的事")
    # 阶段二：立项四问
    sponsor: str = Field("", description="谁说了算（有分量、能拍板的负责人）")
    budget: str = Field("", description="预算情况（真实预算/大致量级/无）")
    success_metric: str = Field("", description="客户认可的成效指标")
    exit_condition: str = Field("", description="退出条件/项目边界（何时算结束）")
    # 阶段三：客户四问
    champion: str = Field("", description="关键引路人（愿意带进业务会议的人）")
    data_access: str = Field("", description="数据与权限开放程度（哪怕只读）")
    feedback_cycle: str = Field("", description="反馈周期（建议给出去多久知道对错）")
    explorer_payer: str = Field("", description="谁为前期探索买单（合同能否容忍不确定期）")
    # 阶段四：真问题挖掘
    time_sinks: str = Field("", description="执行层每天/每周最耗时的重复性工作")
    history_attempts: str = Field("", description="历史上的改进尝试与结果（阻力地图）")
    baseline: str = Field("", description="关键指标基线数字（访谈所得）")
    # 阶段五：场景化
    candidate_scenarios: List[Scenario] = Field(default_factory=list, description="候选场景清单")
    risks: List[str] = Field(default_factory=list, description="主要风险清单")
    not_do: List[str] = Field(default_factory=list, description="显式不建议做的（敢说不）")
    open_questions: List[str] = Field(default_factory=list, description="遗留待确认问题")
    notes: str = Field("", description="其他值得记录的访谈发现")

    def filled(self, *names: str) -> bool:
        for n in names:
            v = getattr(self, n, None)
            if isinstance(v, list):
                if not v:
                    return False
            elif not (v and str(v).strip()):
                return False
        return True


# 阶段计划：(阶段名, 说明, 该阶段需要填的字段, 提问技法提示)
FIELD_PLAN = [
    ("阶段一·客户画像", "破冰，建立基本认知",
     ["industry", "interviewee_role", "top_pain"],
     "像半日免费诊断会开场：不谈技术，先问『今年最想改善的业务指标是什么？现在这个数是多少？改善 10% 值多少钱？』"),
    ("阶段二·立项四问", "筛项目：这事值不值得做",
     ["sponsor", "budget", "success_metric", "exit_condition"],
     "四问逐个确认：谁说了算？有没有真实预算？用什么指标衡量成功？最关键——有没有明确的退出条件？可向客户解释：没有退出条件的项目，不是项目，而是无底洞式的支持合同。"),
    ("阶段三·客户四问", "筛组织：客户接不接得住 FDE 式合作",
     ["champion", "data_access", "feedback_cycle", "explorer_payer"],
     "四问逐个确认：有没有人愿意带你进业务会议？真实数据和权限给不给（哪怕从只读开始）？反馈周期多长？前期探索阶段谁买单？"),
    ("阶段四·真问题挖掘", "越过两层陷阱：嘴上需求与你的想当然",
     ["time_sinks", "history_attempts", "baseline"],
     "别问『你们一般怎么做』，问『上周二下午三点，你手头上在忙什么』；追问每月都要人工补救的『奇怪的星期二』；问过去有没有人试着改进过、结果如何；对每个痛点追问指标和当前数字。"),
    ("阶段五·场景与优先级", "把嘴上的需求翻译成可衡量的结果",
     ["candidate_scenarios", "risks", "not_do"],
     "把痛点翻译成北极星指标（三条件：与商业价值强相关/努力可影响/可拆解可归因），形成候选场景并按价值·可行性·可见性打分，给出 做/延后/放弃 三档结论；明确写出『不做什么』。"),
]

STAGES_DONE = "done"


def next_stage(info: DiagnosisInfo) -> str:
    """返回下一个待推进的阶段名；全部完成返回 STAGES_DONE。"""
    for stage, _, fields, _ in FIELD_PLAN:
        if not info.filled(*fields):
            return stage
    return STAGES_DONE


def stage_progress(info: DiagnosisInfo) -> str:
    lines = []
    for stage, desc, fields, _ in FIELD_PLAN:
        got = sum(1 for f in fields if info.filled(f))
        mark = "✓" if got == len(fields) else ("…" if got else " ")
        lines.append(f"  [{mark}] {stage}（{desc}）{got}/{len(fields)}")
    extra = []
    if info.candidate_scenarios:
        extra.append(f"  候选场景 {len(info.candidate_scenarios)} 个")
    if info.risks:
        extra.append(f"  风险 {len(info.risks)} 条")
    return "\n".join(lines + extra)
