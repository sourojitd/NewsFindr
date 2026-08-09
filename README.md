<!--
  NewsFindr — Agentic AI for Personalized News Retrieval
  Author: Sourojit Dhua
-->

<div align="center">

<!-- Animated typing banner -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3200&pause=900&color=38BDF8&center=true&vCenter=true&multiline=true&width=780&height=100&lines=NewsFindr+%E2%80%94+Agentic+AI+News+Assistant;Personalised+%C2%B7+Fresh+%C2%B7+Credibility-Filtered" alt="NewsFindr typing banner" />

<br/>

<!-- Identity / stack badges -->
<a href="https://github.com/sourojitd">
  <img src="https://img.shields.io/badge/Author-Sourojit%20Dhua-0ea5e9?style=for-the-badge&logo=github&logoColor=white" alt="Author Sourojit Dhua"/>
</a>
<img src="https://img.shields.io/badge/Domain-Advanced%20GenAI%20for%20NLP-8b5cf6?style=for-the-badge&logo=openai&logoColor=white" alt="Advanced GenAI for NLP"/>
<img src="https://img.shields.io/badge/Status-Completed-22c55e?style=for-the-badge&logo=checkmarx&logoColor=white" alt="Completed"/>

<br/><br/>

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/LangChain-Agents%20%26%20Tools-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangChain"/>
<img src="https://img.shields.io/badge/Groq-LLaMA%203.1%208B-F55036?style=flat-square&logo=groq&logoColor=white" alt="Groq"/>
<img src="https://img.shields.io/badge/SQLite-SQL%20Agent-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite"/>
<img src="https://img.shields.io/badge/DuckDuckGo-Live%20Search-DE5833?style=flat-square&logo=duckduckgo&logoColor=white" alt="DuckDuckGo"/>
<img src="https://img.shields.io/badge/ReAct-ZERO__SHOT-111827?style=flat-square" alt="ReAct"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter"/>
<img src="https://img.shields.io/badge/License-Educational-informational?style=flat-square" alt="License"/>

<br/><br/>

**An end-to-end multi-agent system that turns a customer email into a curated, trustworthy news digest — built and documented by [Sourojit Dhua](https://github.com/sourojitd).**

<br/>

<a href="https://sourojitd.github.io/AIML-NewsFindr/">
  <img src="https://img.shields.io/badge/Live_Site-sourojitd.github.io%2FAIML-NewsFindr-0d9488?style=for-the-badge&logo=githubpages&logoColor=white" alt="Live GitHub Pages site"/>
</a>
<a href="https://sourojitd.github.io/AIML-NewsFindr/">
  <img src="https://img.shields.io/badge/Interactive-Pipeline%20%26%20Topics-f59e0b?style=for-the-badge" alt="Interactive showcase"/>
</a>

<br/><br/>

[**Live showcase**](https://sourojitd.github.io/AIML-NewsFindr/) · [Notebook](#-repository-contents) · [Architecture](#-system-architecture) · [AIML Topics](#-aiml-topics-demonstrated) · [Pipeline](#-end-to-end-data-flow) · [Run](#-quick-start)

</div>

---

## Overview

Modern readers drown in volume and starve for signal. **NewsFindr** closes that gap with an **agentic GenAI pipeline**:

1. Verify a customer in a relational CRM (`customer.db`)
2. Expand their interests (+ optional free-text query) into precise news searches
3. Retrieve **fresh** results from the open web (DuckDuckGo, past week)
4. Keep only **trusted publishers**, then LLM-rank for relevance
5. Fetch article text and produce concise per-URL summaries
6. Return a **Top-3** personalised digest

The same capabilities are exposed two ways:

| Mode | Entry point | Why it matters |
|---|---|---|
| **Deterministic pipeline** | `newsfindr()` / `query_response()` | Auditable, reproducible, rate-limit friendly |
| **True ReAct agent** | `initialize_agent` + `ZERO_SHOT_REACT_DESCRIPTION` | Autonomously chooses *when* and *in what order* to call tools |

> Developed as an **Advanced GenAI for NLP** project by **Sourojit Dhua**, covering LLM ops, SQL agents, tool-calling, credibility filtering, and ReAct orchestration end to end.

---

## What problem it solves

```mermaid
flowchart LR
    subgraph Pain["Information overload"]
        A["Generic feeds"] --> B["Noise & outdated links"]
        B --> C["Low trust / high effort"]
    end

    subgraph NewsFindr["NewsFindr solution"]
        D["CRM interests + user query"] --> E["Agentic retrieval"]
        E --> F["Trusted + relevant Top-3"]
        F --> G["LLM summaries"]
    end

    C -.->|replaced by| D
    G --> H["Personalised digest"]

    style Pain fill:#3f1d1d,stroke:#f87171,color:#fecaca
    style NewsFindr fill:#0f2744,stroke:#38bdf8,color:#e0f2fe
    style H fill:#14532d,stroke:#4ade80,color:#dcfce7
```

**Business objectives covered in the notebook**

- Real-time, personalised discovery from verified customer interests
- Credibility via a curated publisher allowlist before the LLM ever judges content
- Reduced overload through freshness filters, deduplication, and Top-3 ranking
- Explainable orchestration (pipeline traces + verbose ReAct tool calls)

---

## System architecture

```mermaid
flowchart TB
    U["Customer email + optional user query"] --> ORCH["NewsFindr Orchestrator"]

    ORCH --> SQL["SQL Agent<br/>LangChain + SQLite + Groq"]
    SQL --> CRM[("customer.db<br/>15 sample customers")]
    SQL --> PROF["Verified profile<br/>name · email · interests"]

    PROF --> SEARCH["Search Interface"]
    SEARCH --> QE["Query Expansion LLM"]
    QE --> DDG["DuckDuckGo Search<br/>timelimit = past week"]
    DDG --> TRUST["Trusted-domain allowlist"]
    TRUST --> REL["LLM Relevance Judge"]

    REL --> OUT["Output / Summariser"]
    OUT --> FETCH["Article fetch<br/>requests + BeautifulSoup"]
    FETCH --> SUM["Groq LLaMA-3.1 summarisation"]
    SUM --> TOP["Top-3 URLs + summaries"]

    ORCH -.->|alternate path| REACT["ReAct Agent<br/>4 Tools · ZERO_SHOT_REACT_DESCRIPTION"]
    REACT --> T1["ExpandSearchQueries"]
    REACT --> T2["DuckDuckGoSearch"]
    REACT --> T3["CredibilityFilter"]
    REACT --> T4["SummarizeNews"]
    T1 --> T2 --> T3 --> T4 --> TOP

    style ORCH fill:#1e3a5f,stroke:#38bdf8,color:#f0f9ff
    style SQL fill:#312e81,stroke:#a5b4fc,color:#eef2ff
    style SEARCH fill:#164e63,stroke:#67e8f9,color:#ecfeff
    style OUT fill:#14532d,stroke:#86efac,color:#f0fdf4
    style REACT fill:#4a044e,stroke:#e879f9,color:#fdf4ff
    style TOP fill:#422006,stroke:#fbbf24,color:#fffbeb
```

### Dual LLM clients (temperature as a design choice)

| Client | Temperature | Role |
|---|---|---|
| `llm` | **0.0** | Deterministic workhorse — SQL, expansion, relevance, summarisation, ReAct |
| `llm_high` | **0.9** | Creative contrast — same prompt yields varied taglines (pedagogical demo) |

Model: **`llama-3.1-8b-instant`** on **Groq** (low latency + tool-calling), with a shared `InMemoryRateLimiter` for free-tier safety.

---

## End-to-end data flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Pipe as newsfindr / query_response
    participant SQL as SQL Agent
    participant DB as customer.db
    participant Exp as Query Expansion
    participant Web as DuckDuckGo
    participant Filter as Trust + Relevance
    participant Sum as Summariser LLM

    User->>Pipe: email + user_query
    Pipe->>SQL: verify & retrieve profile
    SQL->>DB: safe SELECT on customers
    DB-->>SQL: interests JSON
    SQL-->>Pipe: profile dict
    loop for each interest
        Pipe->>Exp: interest + user_query
        Exp-->>Pipe: expanded news query
        Pipe->>Web: search (past week)
        Web-->>Pipe: title / link / snippet
    end
    Pipe->>Filter: domain allowlist → LLM rank
    Filter-->>Pipe: ordered trusted URLs
    Pipe->>Sum: fetch text + summarise
    Sum-->>User: Top-3 digest
```

---

## ReAct agent loop

Four LangChain `Tool`s are bound to a **ZERO_SHOT_REACT_DESCRIPTION** agent. The model plans with Thought → Action → Observation until a Final Answer (or a graceful fallback to the deterministic pipeline).

```mermaid
stateDiagram-v2
    [*] --> Thought: agent receives task
    Thought --> Action: choose a tool
    Action --> ExpandSearchQueries: tool 1
    Action --> DuckDuckGoSearch: tool 2
    Action --> CredibilityFilter: tool 3
    Action --> SummarizeNews: tool 4
    ExpandSearchQueries --> Observation
    DuckDuckGoSearch --> Observation
    CredibilityFilter --> Observation
    SummarizeNews --> Observation
    Observation --> Thought: continue if needed
    Observation --> FinalAnswer: enough evidence
    FinalAnswer --> [*]

    note right of Thought
      handle_parsing_errors = True
      max_iterations = 10
      temperature = 0 (deterministic llm)
    end note
```

---

## AIML topics demonstrated

This repository is structured to show depth across the GenAI / NLP stack — not just “call an API”.

<details open>
<summary><b>Click to expand the skills map</b></summary>

<br/>

| Area | What Sourojit implemented | Where |
|---|---|---|
| **LLM Ops** | Groq `ChatGroq`, dual temperatures, rate limiting, retries, `.env` secrets | §1 |
| **Prompt Engineering** | System prefixes, structured JSON outputs, editor / judge / summariser personas | §2–4 |
| **SQL Agents** | `create_sql_agent`, read-only SELECT constraints, NL → SQL, negative email tests | §2 |
| **Hybrid retrieval** | Deterministic SQL helper + agent path for speed vs explainability | §2.4 |
| **Query understanding** | Interest → expanded news query + free-text `user_query` blending | §3.1 |
| **Web RAG-lite** | DuckDuckGo live search, freshness (`timelimit="w"`), snippet normalisation | §3.2 |
| **Trust & safety** | Curated publisher allowlist before LLM scoring (misinformation control) | §3.3 |
| **LLM-as-judge** | Relevance ranking over candidate articles | §3.4 |
| **Content extraction** | `requests` + BeautifulSoup with snippet fallback | §4.1 |
| **Summarisation** | Per-URL concise digests grounded in fetched text | §4 |
| **Pipeline design** | Dedup by URL/domain, Top-N selection, verbose audit trail | §4.2 |
| **Tool calling** | Four string-tolerant Tools for small-model robustness | §5.1 |
| **ReAct agents** | `initialize_agent`, intermediate steps, parse-error recovery, fallback | §5.2–5.3 |
| **Evaluation demos** | Three customers × distinct queries (Kevin, Julia, Alice) | §6 |
| **Engineering hygiene** | `requirements.txt`, idempotent DB seed, HTML export of executed notebook | repo |

</details>

```mermaid
mindmap
  root((NewsFindr<br/>Sourojit Dhua))
    LLM Foundations
      Groq LLaMA 3.1
      Temperature 0 vs 0.9
      Rate limiting
      Prompt templates
    Agentic Systems
      SQL Agent
      ReAct ZERO_SHOT
      Tool orchestration
      Fallback strategies
    Retrieval & NLP
      Query expansion
      Live web search
      Domain trust filter
      Relevance judge
      Summarisation
    Data & Product
      SQLite CRM
      Personalisation
      Top-3 digests
      Explainable traces
```

---

## Live project site

Interactive public page (pipeline explorer, topic deep-dives, temperature demo, animations):

**https://sourojitd.github.io/AIML-NewsFindr/**

Source lives in [`docs/`](./docs) and is served via GitHub Pages.

## Repository contents

| File | Purpose |
|---|---|
| `NewsFindr.ipynb` | Main executed notebook (primary deliverable) |
| `NewsFindr.html` / `Sourojit_NewsFindr.html` | Static HTML render of the notebook |
| `docs/` | GitHub Pages showcase site |
| `README.md` | Project brief |
| `customer.db` | Input CRM data — SQLite with 15 customers |

### About `customer.db`

**Yes — `customer.db` is the project’s input data file.** You should upload it.

It stores the sample CRM the SQL agent queries:

| Column | Meaning |
|---|---|
| `customer_id` | Stable business key |
| `name` | Display name |
| `email` | Lookup key for personalisation |
| `interests` | JSON list of topics (e.g. Politics, Startups, Travel) |
| `last_updated` | Timestamp |

The notebook can also recreate/seed this DB idempotently (`INSERT OR IGNORE`), but shipping the file makes the demo one-command runnable for reviewers.

---

## Quick start

1. Clone the repo and open `NewsFindr.ipynb` in Jupyter / VS Code.
2. Create a local `.env` with `GROQ_API_KEY=...` (never commit this file).
3. Run all cells — the notebook installs its own dependencies in the setup section.

View the static walkthrough anytime via `Sourojit_NewsFindr.html`, or the interactive site at [sourojitd.github.io/AIML-NewsFindr](https://sourojitd.github.io/AIML-NewsFindr/).

---

## Example queries exercised

| Customer | Interests | User query |
|---|---|---|
| **Kevin** | Politics, Startups, Travel | *latest developments this week* |
| **Julia** | India, Automobile, Business | *electric vehicles and the Indian economy* |
| **Alice** | Politics, Technology, Business | *AI regulation and big tech* |

Each run verifies email → retrieves interests → blends the free-text query → searches → filters → summarises → returns **Top-3**.

---

## Design decisions that show engineering judgment

```mermaid
flowchart LR
    A["Small open model<br/>llama-3.1-8b-instant"] --> B["String-tolerant Tools<br/>plain text OR JSON"]
    A --> C["handle_parsing_errors"]
    A --> D["Deterministic fallback<br/>if ReAct doesn't converge"]
    E["Trust before LLM"] --> F["Allowlist domains"]
    F --> G["Then relevance judge"]
    H["Temperature = 0 for agents"] --> I["Auditable decisions"]
    J["Temperature = 0.9 demo"] --> K["Teach sampling behaviour"]

    style E fill:#14532d,stroke:#86efac,color:#f0fdf4
    style H fill:#1e3a5f,stroke:#38bdf8,color:#f0f9ff
    style A fill:#4a044e,stroke:#e879f9,color:#fdf4ff
```

- **Trust gate before the LLM** — domain allowlist removes unverified publishers cheaply; the model only ranks survivors.
- **Two orchestration styles** — pipeline for production-like reproducibility; ReAct for agentic autonomy.
- **Rate-limit awareness** — shared limiter + retries so free-tier Groq stays usable during demos.
- **Graceful degradation** — article fetch failures fall back to search snippets; agent non-convergence falls back to `newsfindr()`.

---

## Project rubric coverage

| # | Criteria | Status |
|---|---|---|
| 1 | Setting Up the LLM (install, Groq load, simple query, temperature demo) | Completed |
| 2 | SQL Agent for Data Retrieval | Completed |
| 3 | Interface Between SQL and Search Agents | Completed |
| 4 | Output from LLMs (URLs + summaries) | Completed |
| 5 | Creating the Agent (Tools + ReAct) | Completed |
| 6 | Querying with the Agent (3 example users) | Completed |
| 7 | Conclusion | Completed |

---

## Author

<div align="center">

### Sourojit Dhua

**Advanced GenAI · NLP · Agentic Systems · LangChain · Groq**

Built **NewsFindr** as a full-stack demonstration of modern AIML practice:  
relational grounding → live retrieval → trust filtering → tool-using agents → grounded summarisation.

[![GitHub](https://img.shields.io/badge/GitHub-sourojitd-181717?style=for-the-badge&logo=github)](https://github.com/sourojitd)

<sub>If this README or notebook helped you learn agentic NLP patterns, a star is appreciated.</sub>

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0ea5e9,50:6366f1,100:22d3ee&height=120&section=footer&text=NewsFindr%20%C2%B7%20Sourojit%20Dhua&fontSize=18&fontColor=ffffff&animation=fadeIn&fontAlignY=35" alt="footer wave"/>

</div>
