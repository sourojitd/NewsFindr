/* NewsFindr interactive showcase — Sourojit Dhua */

const STEPS = [
  {
    id: "llm",
    name: "LLM Setup",
    chips: ["Groq", "LLaMA 3.1 8B", "Temperature"],
    title: "Reasoning engine on Groq",
    body: "Two ChatGroq clients share one model and a rate limiter. Temperature 0 powers every agentic decision; temperature 0.9 demonstrates creative sampling on the same prompt.",
    points: [
      "Secrets loaded from .env — never hard-coded",
      "InMemoryRateLimiter keeps free-tier calls sustainable",
      "Tool-calling ready model for SQL + ReAct agents",
    ],
  },
  {
    id: "sql",
    name: "SQL Agent",
    chips: ["LangChain", "SQLite", "NL → SQL"],
    title: "Ground personalisation in CRM data",
    body: "A constrained SQL agent verifies customer email against customer.db and returns interests as structured JSON. A deterministic helper sits beside the agent for fast, auditable lookups.",
    points: [
      "create_sql_agent with a read-only system prefix",
      "15 sample customers with interest lists",
      "Negative tests for unknown emails",
    ],
  },
  {
    id: "search",
    name: "Search Interface",
    chips: ["Query expansion", "DuckDuckGo", "Freshness"],
    title: "Turn interests into live news queries",
    body: "Each interest (plus optional free-text user query) expands into a precise search string. DuckDuckGo returns past-week results so the digest stays current.",
    points: [
      "Editor-style prompt for query rewriting",
      "timelimit='w' enforces freshness",
      "Normalised hits: title · link · snippet",
    ],
  },
  {
    id: "trust",
    name: "Trust + Relevance",
    chips: ["Allowlist", "LLM-as-judge", "Safety"],
    title: "Credibility before creativity",
    body: "Untrusted domains are dropped by a curated publisher allowlist. Only survivors reach an LLM relevance judge that ranks articles against the customer profile.",
    points: [
      "Reuters, BBC, TechCrunch, and more",
      "Misinformation control before summarisation",
      "Best-first ordering for Top-N selection",
    ],
  },
  {
    id: "output",
    name: "Summaries",
    chips: ["BeautifulSoup", "Grounded NLG", "Top-3"],
    title: "Fetch, summarise, deliver",
    body: "Article text is fetched when possible, with graceful snippet fallback. Groq produces concise per-URL digests. The pipeline returns a personalised Top-3.",
    points: [
      "URL / domain deduplication",
      "newsfindr() end-to-end orchestrator",
      "query_response() for reproducible demos",
    ],
  },
  {
    id: "react",
    name: "ReAct Agent",
    chips: ["Tools", "ZERO_SHOT", "Autonomy"],
    title: "True agentic orchestration",
    body: "The same four capabilities become LangChain Tools. A ZERO_SHOT_REACT_DESCRIPTION agent decides Thought → Action → Observation, with parse-error recovery and a deterministic fallback.",
    points: [
      "ExpandSearchQueries · DuckDuckGoSearch",
      "CredibilityFilter · SummarizeNews",
      "max_iterations + intermediate_steps recovery",
    ],
  },
];

const TOPICS = [
  {
    cat: "llm",
    title: "LLM Ops on Groq",
    blurb: "Model choice, dual temperatures, retries, and rate limiting.",
    detail:
      "llama-3.1-8b-instant is selected for low latency and tool-calling. A shared InMemoryRateLimiter (~0.5 req/s) plus max_retries keeps demos inside free-tier budgets while still feeling interactive.",
  },
  {
    cat: "llm",
    title: "Prompt engineering",
    blurb: "Personas for editor, SQL guardian, relevance judge, and summariser.",
    detail:
      "ChatPromptTemplate patterns enforce structured outputs (SQL only, JSON indices, concise digests). System prefixes constrain unsafe behaviour before tools run.",
  },
  {
    cat: "agents",
    title: "SQL Agent",
    blurb: "Natural language → safe SELECT against SQLite CRM.",
    detail:
      "create_sql_agent is primed with a CRM-specific system message. Parallel deterministic get_customer_by_email() shows when agents vs helpers are the right abstraction.",
  },
  {
    cat: "agents",
    title: "ReAct tool loop",
    blurb: "Autonomous multi-tool planning with graceful failure modes.",
    detail:
      "initialize_agent + ZERO_SHOT_REACT_DESCRIPTION. Tools accept plain text or JSON so small models stay unblocked. If the loop never emits a Final Answer, SummarizeNews observations or newsfindr() take over.",
  },
  {
    cat: "nlp",
    title: "Query expansion",
    blurb: "Interest tokens become news-seeking search queries.",
    detail:
      "Optional free-text user_query blends with CRM interests so the same customer can steer focus (elections, EVs, AI regulation) without losing personalisation.",
  },
  {
    cat: "nlp",
    title: "Live web retrieval",
    blurb: "DuckDuckGo past-week search as a lightweight RAG source.",
    detail:
      "Hits are normalised and deduplicated. Freshness is a first-class filter — the system is built for current events, not stale index dumps.",
  },
  {
    cat: "nlp",
    title: "Grounded summarisation",
    blurb: "Fetch page text, then summarise with snippet fallback.",
    detail:
      "requests + BeautifulSoup extract readable text (capped). Summaries stay short and URL-linked so reviewers can audit every claim against a source.",
  },
  {
    cat: "safety",
    title: "Trusted-domain allowlist",
    blurb: "Hard gate before any LLM relevance scoring.",
    detail:
      "Engineering judgment: cheap symbolic filters remove dubious publishers first. The LLM only ranks among reputable domains — trust before creativity.",
  },
  {
    cat: "safety",
    title: "LLM-as-judge ranking",
    blurb: "Relevance ordering over candidate articles.",
    detail:
      "Given interests + user request + numbered candidates, the model returns ordered indices only. That keeps the pipeline programmable and easy to inspect.",
  },
  {
    cat: "product",
    title: "Dual orchestration modes",
    blurb: "Deterministic pipeline + autonomous agent, same skills.",
    detail:
      "Pipeline mode is auditable and rate-limit friendly for demos. ReAct mode proves true agency. Shipping both shows product and research maturity.",
  },
  {
    cat: "product",
    title: "Personalised digests",
    blurb: "Email → interests → Top-3 news with summaries.",
    detail:
      "Demo customers Kevin, Julia, and Alice exercise different interest stacks and free-text intents — politics/startups, EVs in India, AI regulation.",
  },
];

const TEMP_LINES = {
  "0": [
    "News that knows you, instantly.",
    "News that knows you, instantly.",
    "News that knows you, instantly.",
  ],
  "0.9": [
    "Your interests. Today's signal.",
    "Curated headlines, zero noise.",
    "Fresh stories, tuned to you.",
  ],
};

function $(sel, root = document) {
  return root.querySelector(sel);
}

function $all(sel, root = document) {
  return [...root.querySelectorAll(sel)];
}

/* Hero signal canvas */
function initHeroCanvas() {
  const canvas = $("#signal-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let raf;
  let t = 0;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function draw() {
    const { width, height } = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, width, height);
    const cx = width * 0.5;
    const cy = height * 0.42;
    const maxR = Math.max(width, height) * 0.55;

    for (let i = 0; i < 6; i++) {
      const phase = (t * 0.35 + i * 0.7) % 1;
      const r = phase * maxR;
      const alpha = (1 - phase) * 0.35;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(45, 212, 191, ${alpha})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // orbiting nodes
    for (let i = 0; i < 18; i++) {
      const a = t * 0.4 + (i / 18) * Math.PI * 2;
      const rr = 80 + (i % 5) * 42;
      const x = cx + Math.cos(a) * rr;
      const y = cy + Math.sin(a * 0.9) * rr * 0.55;
      ctx.beginPath();
      ctx.arc(x, y, 2.2 + (i % 3), 0, Math.PI * 2);
      ctx.fillStyle = i % 3 === 0 ? "rgba(245,158,11,0.85)" : "rgba(153,246,228,0.75)";
      ctx.fill();
    }

    // soft grid
    ctx.strokeStyle = "rgba(153,246,228,0.05)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 48) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 48) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    t += 0.016;
    if (!reduce) raf = requestAnimationFrame(draw);
  }

  resize();
  draw();
  window.addEventListener("resize", () => {
    resize();
    if (reduce) draw();
  });
  window.addEventListener("beforeunload", () => cancelAnimationFrame(raf));
}

/* Pipeline */
function initPipeline() {
  const list = $("#pipeline-steps");
  const panel = $("#pipeline-panel");
  if (!list || !panel) return;

  list.innerHTML = STEPS.map(
    (s, i) => `
    <button class="step-btn${i === 0 ? " is-active" : ""}" type="button" data-step="${s.id}" aria-pressed="${i === 0}">
      <span class="num">0${i + 1}</span><span class="name">${s.name}</span>
    </button>`
  ).join("");

  function render(step) {
    panel.innerHTML = `
      <div class="panel-meta">
        ${step.chips.map((c, i) => `<span class="chip${i === step.chips.length - 1 ? " amber" : ""}">${c}</span>`).join("")}
      </div>
      <h3 class="panel-title">${step.title}</h3>
      <p class="panel-body">${step.body}</p>
      <ul class="panel-list">${step.points.map((p) => `<li>${p}</li>`).join("")}</ul>
    `;
  }

  render(STEPS[0]);

  list.addEventListener("click", (e) => {
    const btn = e.target.closest(".step-btn");
    if (!btn) return;
    $all(".step-btn", list).forEach((b) => {
      b.classList.toggle("is-active", b === btn);
      b.setAttribute("aria-pressed", b === btn ? "true" : "false");
    });
    const step = STEPS.find((s) => s.id === btn.dataset.step);
    render(step);
  });

  // auto-advance gently
  let idx = 0;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce) {
    setInterval(() => {
      if (document.hidden) return;
      idx = (idx + 1) % STEPS.length;
      const btn = list.querySelectorAll(".step-btn")[idx];
      btn?.click();
    }, 5200);
  }
}

/* Topics */
function initTopics() {
  const grid = $("#topic-grid");
  const filters = $("#topic-filters");
  if (!grid || !filters) return;

  const cats = [
    { id: "all", label: "All topics" },
    { id: "llm", label: "LLM Ops" },
    { id: "agents", label: "Agents" },
    { id: "nlp", label: "NLP / Retrieval" },
    { id: "safety", label: "Trust & Safety" },
    { id: "product", label: "Product Design" },
  ];

  filters.innerHTML = cats
    .map(
      (c, i) =>
        `<button type="button" class="filter-btn${i === 0 ? " is-active" : ""}" data-cat="${c.id}">${c.label}</button>`
    )
    .join("");

  function paint(cat) {
    grid.innerHTML = TOPICS.filter((t) => cat === "all" || t.cat === cat)
      .map(
        (t) => `
      <button type="button" class="topic-card" data-cat="${t.cat}">
        <h3>${t.title}</h3>
        <p class="blurb">${t.blurb}</p>
        <div class="detail">${t.detail}</div>
      </button>`
      )
      .join("");
  }

  paint("all");

  filters.addEventListener("click", (e) => {
    const btn = e.target.closest(".filter-btn");
    if (!btn) return;
    $all(".filter-btn", filters).forEach((b) => b.classList.toggle("is-active", b === btn));
    paint(btn.dataset.cat);
  });

  grid.addEventListener("click", (e) => {
    const card = e.target.closest(".topic-card");
    if (!card) return;
    const open = card.classList.contains("is-open");
    $all(".topic-card", grid).forEach((c) => c.classList.remove("is-open"));
    if (!open) card.classList.add("is-open");
  });
}

/* Temperature demo */
function initTempDemo() {
  const out = $("#temp-lines");
  const meter = $("#temp-meter-fill");
  const label = $("#temp-label");
  if (!out) return;

  function show(temp) {
    const lines = TEMP_LINES[temp];
    if (meter) meter.style.width = temp === "0" ? "8%" : "90%";
    if (label) {
      label.textContent =
        temp === "0"
          ? "temperature = 0 → deterministic, repeatable"
          : "temperature = 0.9 → creative, varied";
    }
    out.innerHTML = lines
      .map((line, i) => `<p style="animation-delay:${i * 0.08}s">${i + 1}. ${line}</p>`)
      .join("");
  }

  $all(".temp-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $all(".temp-btn").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      show(btn.dataset.temp);
    });
  });

  show("0");
}

/* Scroll UX */
function initScrollUX() {
  const bar = $("#scroll-progress");
  const nav = $(".nav");

  function onScroll() {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const p = max > 0 ? (window.scrollY / max) * 100 : 0;
    if (bar) bar.style.width = `${p}%`;
    nav?.classList.toggle("is-scrolled", window.scrollY > 12);
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) e.target.classList.add("is-visible");
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
  );
  $all(".reveal").forEach((el) => io.observe(el));
}

function initNav() {
  const toggle = $("#nav-toggle");
  const links = $("#nav-links");
  toggle?.addEventListener("click", () => {
    const open = links.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  });
  links?.addEventListener("click", (e) => {
    if (e.target.tagName === "A") links.classList.remove("is-open");
  });
}

function initCounters() {
  const nums = $all("[data-count]");
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        const el = e.target;
        const target = Number(el.dataset.count);
        const suffix = el.dataset.suffix || "";
        let n = 0;
        const steps = 28;
        const tick = () => {
          n += 1;
          const val = Math.round((target * n) / steps);
          el.textContent = `${val}${suffix}`;
          if (n < steps) requestAnimationFrame(tick);
        };
        tick();
        io.unobserve(el);
      });
    },
    { threshold: 0.5 }
  );
  nums.forEach((el) => io.observe(el));
}

document.addEventListener("DOMContentLoaded", () => {
  initHeroCanvas();
  initPipeline();
  initTopics();
  initTempDemo();
  initScrollUX();
  initNav();
  initCounters();
});
