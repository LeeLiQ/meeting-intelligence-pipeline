# 02 — Sources (curated starting points)

> Deliberately short. Read the primary ones when their concept comes up in a milestone,
> not all at once. Confirm latest versions/links when you open them. Save anything you
> actually use into `notes/` in your own words — don't hoard.

## Agents — primary
- **Anthropic, "Building Effective Agents"** — the clearest taxonomy of workflows vs.
  agents; start here before any framework. (anthropic.com → Research/Engineering.)
- **ReAct: Synergizing Reasoning and Acting in LLMs** — Yao et al., arXiv:2210.03629.
  The canonical reason+act loop behind Milestone 1.
- **OpenAI Agents SDK docs** — provider-native agents, tools, handoffs. (Milestone 2.)
- **LangGraph docs** — nodes + `State` graph; maps onto your `PipelineStage`/`PipelineContext`.
  (Milestone 2.)
- **Your provider's tool/function-calling guide** — OpenAI and Gemini both have one;
  read the one matching the provider you extend first. (Milestone 1.)

## RAG / memory — primary
- **RAG: Retrieval-Augmented Generation** — Lewis et al., arXiv:2005.11401. The origin
  and vocabulary. (Milestone 3.)
- **"Lost in the Middle: How Language Models Use Long Contexts"** — Liu et al.,
  arXiv:2307.03172. Why retrieval beats dumping everything — directly relevant to your
  `InterpretationStage` token-cost TODO. (Milestones 3–4.)
- **Anthropic, "Contextual Retrieval"** — practical chunking/embedding improvements.
  (Milestone 3, after the basics work.)
- **A local vector store's docs** — pick one for Milestone 3: Chroma, FAISS, or
  sqlite-vec. Read just the quickstart + the similarity-search call.
- **An embeddings guide** — your chosen embedder's docs (OpenAI `text-embedding-3-small`
  for an API option, or `sentence-transformers` for fully local).

## Practitioner commentary (optional, for intuition)
- Pinecone Learning Center — readable RAG explainers (vendor, but solid fundamentals).
- Simon Willison's blog — grounded, skeptical takes on agents/tools/LLMs.

## Already in this repo (read these as "sources" too)
- `progresses/20260330-architecture_and_design.md` — your own architecture rationale.
- `progresses/week1-summary.md` — your own first cut at pipeline-vs-agent.
- `helper/pipeline/stages.py` docstring — already name-checks LangGraph/Haystack/Agents SDK.

---

# 中文版

> 专业术语保留英文。刻意保持简短。等某个概念在 milestone 里出现时再读对应主要资料，不要
> 一次性全读。打开时确认最新版本/链接。把你真正用到的东西用自己的话写进 `notes/`——别囤积。

## Agents — 主要
- **Anthropic, "Building Effective Agents"** — 对 workflows vs. agents 最清晰的分类；在碰
  任何框架之前先读它。（anthropic.com → Research/Engineering。）
- **ReAct: Synergizing Reasoning and Acting in LLMs** — Yao 等，arXiv:2210.03629。
  Milestone 1 背后那个经典的 reason+act loop。
- **OpenAI Agents SDK docs** — provider 原生的 agents、tools、handoffs。（Milestone 2。）
- **LangGraph docs** — nodes + `State` 图；对应你的 `PipelineStage`/`PipelineContext`。
  （Milestone 2。）
- **你所用 provider 的 tool/function-calling 指南** — OpenAI 和 Gemini 都有；先读你最先
  扩展的那个 provider 对应的。（Milestone 1。）

## RAG / memory — 主要
- **RAG: Retrieval-Augmented Generation** — Lewis 等，arXiv:2005.11401。这个概念和术语的
  源头。（Milestone 3。）
- **"Lost in the Middle: How Language Models Use Long Contexts"** — Liu 等，
  arXiv:2307.03172。为什么 retrieval 胜过全塞——直接关系到你 `InterpretationStage` 的
  token 成本 TODO。（Milestone 3–4。）
- **Anthropic, "Contextual Retrieval"** — 实用的 chunking/embedding 改进。（Milestone 3，
  在基础跑通之后。）
- **某个本地 vector store 的文档** — Milestone 3 选一个：Chroma、FAISS 或 sqlite-vec。只读
  quickstart + similarity-search 调用即可。
- **一份 embeddings 指南** — 你所选 embedder 的文档（API 选项可用 OpenAI
  `text-embedding-3-small`，或全本地用 `sentence-transformers`）。

## 实践者评论（可选，建立直觉）
- Pinecone Learning Center — 易读的 RAG 讲解（厂商出品，但基础扎实）。
- Simon Willison 的博客 — 关于 agents/tools/LLM 的接地气、带批判性的看法。

## 本 repo 里已有的（也当作 "sources" 来读）
- `progresses/20260330-architecture_and_design.md` — 你自己的架构思路。
- `progresses/week1-summary.md` — 你自己对 pipeline-vs-agent 的第一次尝试。
- `helper/pipeline/stages.py` docstring — 已经点名 LangGraph/Haystack/Agents SDK。
