# FDE 售前智能体（LangChain + RAG）

一个能**主动采访客户做售前诊断**、并**基于 FDE 方法论知识库回答具体问题**的智能体应用。
知识库默认收录（缺哪份自动跳过，可在 `config.py` / `.env` 增删路径）：

| 知识源 | 内容 | 说明 |
|---|---|---|
| FDE 课程 SOP | 《FDE 课程关键经验与客户洞察：项目全生命周期 SOP》 | 项目内置副本 `knowledge/` |
| FDE 全景研究报告 | 《模式、案例与落地操作手册》（15.4 万字） | 优先用 `FDE研究报告\md` 章节源，无则解析 PDF |
| FDE 实战课 | 极客时间《FDE 实战课》23 讲课程稿 | 若本机存在 `FDE-PPT\md` 则自动收录 |

---

## 快速开始

```bash
# 1) 安装依赖（Python ≥ 3.10，已在 3.13 验证）
pip install -r requirements.txt

# 2) 配置模型提供方（任选一家 OpenAI 兼容端点）
copy .env.example .env   # 然后编辑 .env 填入 API Key

# 3) 构建知识库（一次性；换知识源或换 embedding 模型后需重跑）
python cli.py ingest

# 4) 启动 Web 应用（推荐）
streamlit run app.py        # 或直接双击 启动Web应用.bat → http://localhost:8501

# 5) 命令行方式（可选）
python cli.py ask           # 问答模式：客户问，顾问答
python cli.py diagnose      # 诊断模式：智能体主动访谈客户（你），走完五阶段出《售前诊断报告》

# 免 API Key 的离线端到端自检
python cli.py selftest
```

## Web 应用（app.py · Streamlit）

双栏布局，侧栏切模式：

- **🩺 诊断访谈**：聊天界面中智能体按五阶段主动提问；侧栏实时显示访谈进度（各阶段字段采集
  ✓/…/未开始）；「生成诊断报告」按钮一键产出 Markdown 报告并可下载；「重新访谈」重置。
- **💬 方法论问答**：RAG 顾问回答，每条回答下方可展开「📚 引用来源」查看命中片段的
  文档/章节/页码与原文摘录；「清空对话」重置会话。

当前 `.env` 已接入 DeepSeek（`deepseek-chat`）+ 本地向量模型
（fastembed + `BAAI/bge-small-zh-v1.5`，DeepSeek 无 embedding 接口故走本地，
模型缓存于 `data/models/`，检索为真实语义向量）。

## 云端部署（Streamlit Community Cloud）

免费部署完整版（含本地向量模型）。仓库需为 public：

1. 打开 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 登录；
2. **New app** → Repository 选 `CoheeYang/FDE-Agent`，Branch 选 `main`，Main file path 填 `app.py`；
3. **Advanced settings** → Python version 选 `3.13`，Secrets 填入：

   ```toml
   FDE_LLM_BASE_URL = "https://api.deepseek.com/v1"
   FDE_LLM_API_KEY = "sk-你的Key"
   FDE_LLM_MODEL = "deepseek-chat"
   FDE_EMBEDDING_PROVIDER = "local"
   HF_ENDPOINT = "https://huggingface.co"
   ```

   > `HF_ENDPOINT` 必填：覆盖代码里为国内网络预设的镜像，Streamlit Cloud 服务器直连
   > huggingface.co 更快更稳。Secrets 会以环境变量注入，优先级高于 `.env`。

4. Deploy。向量库 `data/vectorstore.json` 已随仓库提供，无需重新 ingest；
   首次启动会自动下载 BGE 向量模型（约 90MB），冷启动约 1–2 分钟属正常现象。

## 两种能力（对应需求）

### 能力一：结合客户痛点做诊断（主动提问）

`python cli.py diagnose` 中，智能体不是被动应答，而是按 FDE 方法论**推动一场结构化访谈**，
五阶段全部来自课程与报告的原始方法论：

| 阶段 | 目标 | 方法论出处 |
|---|---|---|
| 一·客户画像 | 行业/角色/最想改善的指标 | 报告 7.1「诊断前问卷·决策人三问」 |
| 二·立项四问 | 谁说了算？预算？成功指标？**退出条件？** | 课程 05 讲 / SOP §2.3 |
| 三·客户四问 | 引路人？数据权限？反馈周期？谁为探索买单？ | 课程 05 讲 / SOP §2.3 |
| 四·真问题挖掘 | 问「上周二下午三点在忙什么」、挖「奇怪的星期二」、问历史尝试 | 课程 05 讲五天摸底五动作 |
| 五·场景与优先级 | 翻译成北极星指标（三条件），候选场景三维评分 → **做/延后/放弃** | 课程 06 讲 / 报告 7.2 decomposition |

- **会话内命令**：`/status` 查看五阶段进度、`/report` 随时出报告、`/quit` 退出。
- **产出**：《售前诊断报告》Markdown，自动保存到 `data/reports/`，结构固定为
  画像摘要 → 立项与组织体检 → 真问题清单 → 候选场景表 → 风险清单 → **显式不做清单** →
  两周行动计划 → 遗留待确认问题（对应报告 7.1 节「诊断汇报」的三件输出 + SOP 的敢说不原则）。

架构上采用**确定性状态机 + LLM 双层设计**：Python 侧跟踪「哪些字段已采集、下一阶段问什么」，
LLM 只负责自然语言提问与信息抽取（结构化输出回填 Pydantic 状态）。这保证了访谈**永远不会漏问**，
也不会被客户带偏节奏——这是纯 prompt 驱动 agent 做不到的。

### 能力二：客户问具体问题时给出指导（RAG）

`python cli.py ask` 中，每个回答都会：

1. **检索**知识库（向量召回 top-5，含文档名/章节/页码元数据）；
2. **结论先行**作答，句末标注出处，如（依据：FDE全景研究报告…｜第七章）；
3. 遵守方法论约束：对方要方案但没给指标/基线时**先反问指标**（「在界定范围之前就跳向方案，
   是一号秒拒」）；场景不该做时**明确说不**并给替代路径；
4. 知识库没有的内容，明说「资料库中没有直接覆盖」，一般性建议单独标注，**绝不编造**；
5. 必要时可自主追加检索（LangChain tool-calling 循环，最多 3 轮）。

## 项目结构

```
FDE-Agent/
├── app.py                 # Streamlit Web 应用（双模式聊天/进度侧栏/报告下载/引用展开）
├── cli.py                 # 命令行入口：ingest / ask / diagnose / selftest
├── config.py              # 提供方预设（智谱/DeepSeek/百炼/Moonshot/OpenAI/本地vLLM）+ 知识源路径
├── .env / .env.example    # 运行配置（已接入 DeepSeek + 本地 BGE 向量）
├── requirements.txt
├── 启动Web应用.bat        # 双击启动 → http://localhost:8501
├── knowledge/             # 内置知识源（FDE 课程 SOP 全文）
├── fde_agent/
│   ├── llm.py             # ChatOpenAI/OpenAIEmbeddings 工厂；本地 BGE 向量；离线哈希 + 脚本化模型
│   ├── knowledge.py       # 章节感知分块（md 标题树 / PDF 章节页码）→ InMemoryVectorStore → dump/load
│   ├── schemas.py         # DiagnosisInfo（立项四问+客户四问+场景结构）与阶段计划
│   ├── prompts.py         # FDE 方法论编码：访谈纪律/提问技法/回答规范/敢说不
│   ├── diagnose.py        # 诊断智能体（状态机 + 结构化抽取 + 报告生成）
│   └── advisor.py         # RAG 问答顾问（检索 + 工具调用循环）
├── smoke_real.py          # 真实链路冒烟（检索质量/问答/访谈）
├── apptest.py             # Streamlit AppTest 无头验证
└── data/
    ├── vectorstore.json   # 持久化向量库
    ├── models/            # 本地向量模型缓存（bge-small-zh）
    └── reports/           # 生成的诊断报告
```

## 配置说明

支持任何 OpenAI 兼容端点。最简配置（.env）：

```ini
FDE_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
FDE_LLM_API_KEY=xxx
FDE_LLM_MODEL=glm-4.7
FDE_EMBEDDING_MODEL=embedding-3
```

只填 `ZHIPUAI_API_KEY` / `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` 等也能自动识别
（DeepSeek/Moonshot 无 embedding 接口，需用 `FDE_EMBEDDING_*` 变量另配一家的 embedding）。

## 「100% 替代售前」的诚实边界

本智能体把售前中**可标准化**的部分做到了自动化：诊断访谈五阶段、方法论问答、诊断报告生成。
但按课程与报告自身的结论，以下部分无法由它独立承担（也是设计上有意不做的）：

1. **责任承担**：FDE 的灵魂是「对业务结果负责」——AI 无法为合同与结果背书，签约与对账必须有人；
2. **现场信任**：「信任是人一次次当面挣来的」（课程 07 讲），干系人经营、饭桌上的真话属于人；
3. **私域知识**：本知识库是通用 FDE 方法论；要接你们自己的产品参数、报价口径、历史案例，
   把对应文档加入 `config.py` 的知识源清单后重跑 `ingest` 即可（这就是报告 12.4 节的「知识库四组件」）。

定位建议：把它用作售前团队的**第一线诊断与培训武器**——自动完成客户初筛访谈与问题翻译，
把人的时间留给高信号环节（现场走查、decomp 会、方案对齐）。这正是课程 10 讲
「AI 是放大器，不是决策者」的落地形态。

## 扩展路线（按需）

- **接入真实产品知识**：在 `config.py` 增加你们的产品手册/案例库路径 → `python cli.py ingest`；
- **Web 界面**：`fde_agent.diagnose.DiagnoseAgent` 与 `fde_agent.advisor.FDEAdvisor` 是纯 Python 类，
  可直接套 FastAPI/Streamlit；
- **访谈记录持久化**：`DiagnoseAgent.info` 是 Pydantic 对象，`model_dump_json()` 即可入库续访；
- **评测**：按课程 11 讲「评估驱动开发」，把 20 组真实客户问题做成黄金集，回归每次 prompt 修改。
