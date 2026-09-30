# -*- coding: utf-8 -*-
"""FDE 售前智能体命令行入口。

用法：
  python cli.py ingest              构建知识库（需已配置 Embedding）
  python cli.py ingest --offline    离线自检式构建（哈希向量）
  python cli.py ask                 问答模式：客户问，顾问答（RAG）
  python cli.py diagnose            诊断模式：智能体主动访谈客户并出报告
  python cli.py selftest            离线端到端自检（无需 API Key）
"""
import argparse
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config as C  # noqa: E402
from fde_agent.knowledge import ingest, load_vectorstore  # noqa: E402
from fde_agent.llm import get_embeddings, get_llm  # noqa: E402


def _bootstrap(offline: bool = False):
    if offline:
        import os
        os.environ["FDE_OFFLINE"] = "1"
    cfg = C.Config()
    return cfg


def _require_embeddings(cfg):
    """生产模式下必须有可用 Embedding（API 配置或本地模型），避免静默退化成哈希向量库。"""
    ok = cfg.offline or cfg.embeddings_ready or cfg.emb_provider == "local"
    if not ok:
        sys.exit("[中止] 未配置 Embedding（FDE_EMBEDDING_* 或 FDE_EMBEDDING_PROVIDER=local）。"
                 "请在 .env 中配置后重试；仅想自检管线可加 --offline 或运行 selftest。")


def cmd_ingest(args):
    cfg = _bootstrap(args.offline)
    _require_embeddings(cfg)
    embeddings = get_embeddings(cfg)
    stats = ingest(cfg, embeddings)
    print(f"知识库构建完成：共 {stats['chunks']} 个分块，已保存到 {C.VECTORSTORE_PATH}")
    for doc, n in sorted(stats["per_doc"].items(), key=lambda x: -x[1]):
        print(f"  {n:>5}  {doc}")


def cmd_ask(args):
    cfg = _bootstrap(args.offline)
    _require_embeddings(cfg)
    embeddings = get_embeddings(cfg)
    store = load_vectorstore(cfg, embeddings)
    llm = get_llm(cfg, temperature=0.3)
    from fde_agent.advisor import FDEAdvisor
    advisor = FDEAdvisor(llm, store)
    history = []
    print("== FDE 问答顾问（输入问题提问，/quit 退出）==")
    print("提示：可以问方法论与场景问题，例如：")
    print("  · 我们想做设备预测性维护，第一步该做什么？")
    print("  · 客户只有 50 万预算想做全厂 AI，我该怎么谈？")
    print("  · 怎么给 AI 项目定义验收标准？")
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        try:
            a = advisor.answer(q, history=history)
        except Exception as e:
            a = f"[出错] {e}"
        history.append(("用户", q))
        history.append(("顾问", a))
        print(f"\n顾问> {a}")


def cmd_diagnose(args):
    cfg = _bootstrap(args.offline)
    _require_embeddings(cfg)
    embeddings = get_embeddings(cfg)
    store = load_vectorstore(cfg, embeddings)
    llm = get_llm(cfg, temperature=0.5)
    from fde_agent.diagnose import SLASH_HELP, DiagnoseAgent
    agent = DiagnoseAgent(llm, store)
    print("== FDE 售前诊断访谈 ==")
    print(SLASH_HELP)
    print("智能体将扮演 FDE 顾问，主动采访客户（你），按方法论推进五个阶段，")
    print("随时 /status 查看进度，完成或 /report 生成诊断报告。\n")
    opening = ("你好，我是 FDE 售前诊断顾问。今天不聊技术，就想搞清楚三件事："
               "钱从哪里漏、人在哪里耗、数据躺在哪里。可以先介绍一下你们是做什么行业的，"
               "以及你自己在里面负责哪块吗？")
    print(f"顾问> {opening}\n")
    agent.history.append(("顾问", opening))
    while True:
        try:
            u = input("客户> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u:
            continue
        if u in ("/quit", "/exit"):
            break
        try:
            reply = agent.turn(u)
        except Exception as e:
            reply = f"[出错] {e}"
        print(f"\n顾问> {reply}\n")
        if agent.is_complete() and getattr(agent, "info", None) is not None:
            if input("\n（五阶段信息已齐备）现在生成诊断报告？[Y/n] ").strip().lower() in ("", "y", "yes"):
                print(agent.generate_report())
            break


def cmd_selftest(args):
    """离线端到端自检：哈希向量 + 脚本化模型，不需要任何 API Key。"""
    import os
    os.environ["FDE_OFFLINE"] = "1"
    cfg = C.Config()
    embeddings = get_embeddings(cfg)
    print("[1/4] 构建知识库（离线哈希向量）...")
    stats = ingest(cfg, embeddings)
    assert stats["chunks"] > 100, "分块数异常"
    print(f"      分块 {stats['chunks']} 个；来源分布：")
    for doc, n in sorted(stats["per_doc"].items(), key=lambda x: -x[1])[:6]:
        print(f"        {n:>5}  {doc}")

    print("[2/4] 加载向量库并检索...")
    store = load_vectorstore(cfg, embeddings)
    for q in ("干系人地图 四问", "北极星指标 三条件", "上线检查清单 六项"):
        hits = store.similarity_search(q, k=2)
        assert hits, f"检索无命中：{q}"
        print(f"      「{q}」→ {hits[0].metadata.get('doc', '?')}｜{hits[0].metadata.get('section', '')[:40]}")

    print("[3/4] 诊断访谈智能体（脚本化模型）...")
    from fde_agent.diagnose import DiagnoseAgent, _ReportOutput as _ReportOut, \
        _TurnOutput as _TurnOut
    from fde_agent.llm import ScriptedChatModel
    from fde_agent.schemas import DiagnosisInfo, Scenario

    def T(reply, **kw):
        return ("structured", _TurnOut(reply=reply, **kw))

    script = [
        T("好的，医疗器械注塑厂，我管生产。最痛的是报废率 3% 和旺季加班。",
          industry="医疗器械注塑件生产，约400人", interviewee_role="生产厂长",
          top_pain="报废率约3%偏高；旺季人工成本涨"),
        T("立项的事我回去确认，预算大概几十万。指标就盯报废率。",
          sponsor="G总（总经理）", budget="几十万级，待最终确认",
          success_metric="注塑件报废率（财务口径月均）",
          exit_condition="先做一个季度，交出预警日报并完成一次对账"),
        T("老张（质检组长）能带我进车间会议；数据可以给只读权限；反馈大概一两周；探索期费用从技改预算走。",
          champion="质检组长（可带进业务会议）",
          data_access="可开只读权限（注塑机参数、ERP订单、纸质报废记录需电子化）",
          feedback_cycle="一至两周",
          explorer_payer="客户技改预算承担前期探索"),
        T("每周最耗时间的是月底汇总纸质质检记录，排产靠主任Excel。",
          time_sinks="质检纸质记录月底汇总；排产靠Excel人工排",
          baseline="报废率3%上下（质检抽检口径）"),
        T("之前找过软件公司做过报表系统，没用起来。",
          history_attempts="做过报表系统，未用起来（阻力：一线填报负担）"),
        T("场景就按质量预警来吧，风险我担心一线不填数。设备联网改造先别碰。",
          candidate_scenarios=[Scenario(
              name="注塑质量预警（最小版）", metric="注塑件报废率（财务口径月均）",
              baseline="3.1%（待财务核）", value_estimate="降0.5个千分点年化约40万",
              data_feasibility="注塑机参数可取；温湿度需加装记录仪", time_to_value="6-8周",
              score="价值4 可行性3 可见性4", verdict="做")],
          risks=["一线填报负担导致数据不真实"],
          not_do=["全厂设备联网改造（一期不做）"]),
    ]
    agent = DiagnoseAgent(ScriptedChatModel(script=script), store)
    for u in ("我们是注塑厂", "预算不多", "数据能开只读", "每周月底最烦", "之前做过没成", "先做质量预警吧"):
        agent.turn(u)
    report_call = ("structured", _ReportOut(markdown="# 售前诊断报告：某注塑厂\n\n"
                                            "## 4. 候选场景\n\n建议：做（依据：FDE课程SOP）"))
    agent.llm.script.append(report_call)
    report = agent.generate_report()
    assert "售前诊断报告" in report and agent.is_complete(), "诊断状态机未走完"
    print(f"      五阶段完成，报告已生成：{agent.report_path}")

    print("[4/4] RAG 问答顾问（脚本化模型）...")
    from fde_agent.advisor import FDEAdvisor
    ans = ("结论：先做数据审计再谈模型。步骤：……（依据：FDE实战课·05）[1]")
    advisor = FDEAdvisor(ScriptedChatModel(script=[("text", ans)]), store)
    out = advisor.answer("第一个项目选什么场景？")
    assert "依据" in out, "问答缺少出处标注"
    print(f"      问答示例：{out[:50]}…")
    print("\nSELFTEST PASS ✅  （管线全通：摄取→检索→诊断状态机→报告→问答）")


def main():
    p = argparse.ArgumentParser(description="FDE 售前智能体（LangChain + RAG）")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "ask", "diagnose"):
        sp = sub.add_parser(name)
        sp.add_argument("--offline", action="store_true", help="离线模式（哈希向量/自检）")
    sub.add_parser("selftest")
    args = p.parse_args()
    {"ingest": cmd_ingest, "ask": cmd_ask, "diagnose": cmd_diagnose,
     "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
