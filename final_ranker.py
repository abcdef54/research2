import os
import re
import math
import time
import random
import asyncio
import hashlib
import dotenv
from functools import lru_cache
from uuid import uuid4
from typing import TypedDict, Annotated, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

dotenv.load_dotenv()

# DEFAULT max candidates compared in a single tournament ranking call — the group size K. One-shot
# ranking degrades as the candidate count grows and K=3 has been the ranker's sweet spot, so by
# default each tournament round ranks groups of 3 and byes the remainder (minimum rank calls).
# This is only the FALLBACK. The effective K is read per run from
# config["configurable"]["max_rank_group"] (harness flag --max-group), so sweeping K for the
# ablation needs no code edit and every other selector is left untouched.
MAX_RANK_GROUP = 3

# Smallest legal K. At K=1 every group is a singleton, i.e. a bye, so no candidate is ever
# eliminated and `while len(survivors) > 1` in `tournament_rank` never exits. Rejected up front
# (see `_resolve_max_group`) instead of hanging a multi-hour benchmark run.
MIN_RANK_GROUP = 2

# How many per-token alternatives to request from the server during GENERATION, used only by the
# `self_certainty_proxy` selector. 20 is the OpenAI-compatible API's documented ceiling for
# `top_logprobs`, so it is the most tail we can see through llama-server's /v1 endpoint. The exact
# self-certainty of Kang et al. sums over the FULL vocabulary; we can only ever see the top slice
# here, which is precisely why this metric is named a proxy (see `_self_certainty_proxy`).
GEN_TOP_LOGPROBS = 20

# ───────────────────────── LLM access ─────────────────────────

@lru_cache(maxsize=8)
def _build_llm(model_name: str, base_url: str, temperature: float) -> ChatOpenAI:
    """Cached so repeated nodes reuse one client. Args are hashable on purpose.

    NOTE: no fixed seed here. A fixed seed (the old seed=42) makes identical prompts return
    identical text, which collapses pass@N -> pass@1. Sampling pins a FRESH random seed per call
    via `_seeded` — only meaningful at temperature > 0 (at temp 0 decoding is greedy/deterministic
    and the seed has no effect).
    """
    return ChatOpenAI(
        base_url=base_url,
        api_key="not-needed",
        model=model_name,
        temperature=temperature,
    )


def get_llm(
    config: RunnableConfig,
    temperature: Optional[float] = None,
) -> ChatOpenAI:
    """Build (or reuse) the local llama.cpp-backed client.
    - model_name comes from config; base_url from env (defaults to the local llama-server).
    """
    cfg = config["configurable"]
    model_name = cfg.get("model_name", "qwen")
    base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:18000/v1")
    temp = temperature if temperature is not None else cfg.get("temperature", 0.0)

    llm = _build_llm(model_name, base_url, temp)

    return llm


def _seeded(llm: ChatOpenAI, seed: int) -> ChatOpenAI:
    """Return a copy of the cached client pinned to a fresh `seed`, sharing the same underlying
    HTTP client (verified: model_copy does not rebuild the connection pool). Distinct seeds make
    the N samples genuinely diverse instead of duplicates (only at temperature > 0)."""
    return llm.model_copy(update={"seed": seed})


# ───────────────────────── Answer extraction / comparison ─────────────────────────

def _sample_idx(state: "AgentState", config: RunnableConfig) -> str:
    """Per-sample tag for log lines, so logs from concurrently-running GSM8K questions don't
    interleave into something that looks impossible. Prefers config['configurable']['sample_idx']
    (set by the eval harness); otherwise falls back to a short stable hash of the question so every
    log line within one invocation shares a tag even before the eval wires sample_idx. Logging only:
    not stored in state, not a metric."""
    cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
    idx = cfg.get("sample_idx")
    if idx is not None:
        return str(idx)
    q = state.get("user_query") or ""
    return hashlib.sha1(q.encode("utf-8")).hexdigest()[:6] if q else "?"


# ── TEMPORARY token-usage debug logging (console only; nothing stored in state/CSV/metrics) ──
# Remove this block when context-usage inspection is no longer needed.

def _estimate_tokens(text: str) -> int:
    """Approximate token count when the API doesn't report usage. Prefers tiktoken; falls back to a
    ~4-chars/token heuristic. Estimate only — used purely for the temporary debug print."""
    if not text:
        return 0
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)


def _token_usage_from_message(msg) -> Optional[tuple]:
    """(prompt, completion, total) from an LLM response message if the API exposed usage, else None.
    Checks usage_metadata first, then response_metadata['token_usage'] / ['usage']."""
    if msg is None:
        return None
    um = getattr(msg, "usage_metadata", None)
    if um:
        p, c, t = um.get("input_tokens"), um.get("output_tokens"), um.get("total_tokens")
        if p is not None or c is not None or t is not None:
            p, c = p or 0, c or 0
            return p, c, (t if t is not None else p + c)
    rm = getattr(msg, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage")
    if tu:
        p = tu.get("prompt_tokens", tu.get("input_tokens"))
        c = tu.get("completion_tokens", tu.get("output_tokens"))
        t = tu.get("total_tokens")
        if p is not None or c is not None or t is not None:
            p, c = p or 0, c or 0
            return p, c, (t if t is not None else p + c)
    return None


def _log_token_usage(idx: str, label: str, msg=None, prompt_text: str = "", completion_text: str = "") -> None:
    """TEMPORARY: print prompt/completion/total tokens for one LLM call. Uses real usage metadata
    when available, otherwise an approximate tokenizer estimate. Console only — stores nothing."""
    usage = _token_usage_from_message(msg)
    if usage is not None:
        p, c, t = usage
    else:
        p = _estimate_tokens(prompt_text)
        c = _estimate_tokens(completion_text)
        t = p + c
    print(f"[IDX {idx}] {label}")
    print(f"Prompt Tokens: {p}")
    print(f"Completion Tokens: {c}")
    print(f"Total Tokens: {t}\n")


# ───────────────────────── Data structures ─────────────────────────

class Candidate(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)  # stable identity for logging
    answer: str = ""
    # Truncated self-certainty proxy computed from the generation call's top-k logprobs (see
    # `_self_certainty_proxy`). None when the server returned no logprobs — the selector treats that
    # as a hard error rather than guessing, so a misconfigured run can't masquerade as a valid one.
    confidence: Optional[float] = None


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str

    reasoning_mode: Literal[
        "baseline",
        "low",
        "medium",
        "high",
        "extra",
        "max"
    ]

    width: int
    pool: list[Candidate]

    final_answer: Optional[str]
    sampled_answers: list[str]

    rank_latency: float
    rank_calls: int
    winner_candidate_id: Optional[str]
    rank_parse_failed: bool

    tournament_rounds: int
    tournament_rank_calls: int
    tournament_max_group_size: int


# ───────────────────────── Mode / difficulty config ─────────────────────────

# Tournament ranking uses ODD widths (1/3/5/7) so each round ranks floor(n/3) groups of 3 and byes
# the trailing n%3, dropping the survivor count straight to <=3 (e.g. 5->[3]+2 byes, 7->[3,3]+1 bye,
# 9->[3,3,3]).
TOURNAMENT_MODE_CONFIG = {
    "baseline": 1,   # width 1 -> 0 ranking calls
    "low":      1,   # width 1 -> 0 ranking calls
    "medium":   3,   # [3] one-shot (1 call)
    "high":     5,   # [3] + 2 byes, then 3-way final (2 calls)
    "extra":    7,   # [3,3] + 1 bye, then 3-way final (3 calls)
}


def get_config(mode: str) -> int:
    return TOURNAMENT_MODE_CONFIG[mode]

# ───────────────────────── Nodes ─────────────────────────

def setup(state: AgentState, config: RunnableConfig) -> dict:
    """Pure-Python entry (NO llm call). Reads mode + governor settings, seeds the run state.
    RETURNS: user_query, reasoning_mode, width, pool."""
    cfg = config["configurable"]
    mode = cfg["reasoning_mode"]
    loop_cfg = get_config(mode)

    return {
        "user_query": state["messages"][-1].content,
        "reasoning_mode": mode,
        "width": loop_cfg,
        "pool": [],
    }


def _self_certainty_proxy(msg) -> Optional[float]:
    """Mean per-token KL divergence from a uniform distribution, computed over the top-k
    alternatives the server returned for each generated token. Higher = the model was more decisive
    about its own tokens.

    Per position i with the k returned alternatives renormalized to q (so they sum to 1):
        KL(U_k || q) = -log k - (1/k) * sum_j log q_j
    and the score is the mean over all positions.

    WHY THIS IS A PROXY, NOT self-certainty: Kang et al. sum over the ENTIRE vocabulary (~150k for
    Qwen). That sum is dominated by the low-probability tail, which no OpenAI-compatible endpoint
    exposes (top_logprobs caps at 20). Truncating to the top-k therefore does NOT approximate their
    number — it measures peakedness over the visible head instead. It is a legitimate confidence
    signal and a legitimate baseline, but it must be reported under its own name; calling it
    "self-certainty" would misattribute the formula. Getting the exact metric requires full logits,
    i.e. running the model in-process (llama-cpp-python) rather than behind llama-server.

    RETURNS: the score, or None when the response carried no usable logprobs (server not configured
    for them) — callers must treat None as an error, never as a low score."""
    rm = getattr(msg, "response_metadata", None) or {}
    content = (rm.get("logprobs") or {}).get("content") or []

    per_token: list[float] = []
    for entry in content:
        alts = entry.get("top_logprobs") or []
        logprobs = [a["logprob"] for a in alts if a.get("logprob") is not None]
        if not logprobs:
            continue                     # position carried no alternatives; skip it
        k = len(logprobs)
        total_p = sum(math.exp(lp) for lp in logprobs)
        if total_p <= 0.0:
            continue
        log_total = math.log(total_p)
        # (1/k) * sum_j log q_j  where  log q_j = logprob_j - log(sum_p)
        mean_log_q = sum(logprobs) / k - log_total
        per_token.append(-math.log(k) - mean_log_q)

    if not per_token:
        return None
    return sum(per_token) / len(per_token)


async def generate(state: AgentState, config: RunnableConfig) -> dict:
    """Self-consistency sampling. Produce `width` INDEPENDENT samples of the RAW user question:
    HumanMessage(s) only, NO system prompt, NO structured output, plain text. Each sample uses a
    fresh random seed (only meaningful at temperature > 0; see MODE_CONFIG note).
    RETURNS: pool."""
    width = state["width"]
    llm = get_llm(config)  # plain client; temperature from config (0.0 for low, >0 for wider modes)

    # Logprobs are requested ONLY for the self_certainty_proxy selector, which scores candidates from
    # them. Other selectors never read them, so their request payloads stay exactly as before and
    # previously-collected majority/tournament numbers remain reproducible.
    want_logprobs = config["configurable"].get("rank_mode") == "self_certainty_proxy"

    idx = _sample_idx(state, config)
    print(f"\n[IDX {idx}] ==================== Generate Node (Self-Consistency) ====================\n")
    print(f"[IDX {idx}] Question: {state['user_query']}")
    print(f"[IDX {idx}] Width (N samples): {width}")
    if want_logprobs:
        print(f"[IDX {idx}] Requesting top_logprobs={GEN_TOP_LOGPROBS} (self-certainty proxy)")

    # Raw question ONLY — no SystemMessage, no system prompt, no structured output.
    prompts = [state["messages"] for _ in range(width)]

    def _client(seed: int):
        c = _seeded(llm, seed)
        return c.bind(logprobs=True, top_logprobs=GEN_TOP_LOGPROBS) if want_logprobs else c

    responses = await asyncio.gather(*[
        _client(random.randint(0, 2**31 - 1)).ainvoke(p) for p in prompts
    ])

    pool = []
    _gen_prompt_text = " ".join(
        m.content for m in state["messages"] if isinstance(getattr(m, "content", None), str)
    )
    for i, r in enumerate(responses):
        content = r.content if isinstance(r.content, str) else str(r.content)
        confidence = _self_certainty_proxy(r) if want_logprobs else None
        pool.append(Candidate(answer=content, confidence=confidence))
        print(f"[IDX {idx}] Sample {i}: id={pool[-1].id[:8]}")
        if want_logprobs:
            shown = f"{confidence:.4f}" if confidence is not None else "MISSING (server returned no logprobs)"
            print(f"[IDX {idx}] Sample {i} self-certainty proxy: {shown}")
        # TEMPORARY token-usage debug (one block per generation call)
        _log_token_usage(idx, "Generate Call", msg=r, prompt_text=_gen_prompt_text, completion_text=content)
    print(f"\n[IDX {idx}] ==========================================================================\n")

    return {
        "pool": pool,
    }

# ───────────────────── One-shot ranking selector ─────────────────────
# A DOMAIN-GENERAL selector. Unlike majority vote — which can only tally when candidates share a
# normalizable answer (identical GSM8K final numbers) and is therefore undefined for open-ended QA,
# summarization, code generation, RAG, research agents, or essay generation — an LLM ranker compares
# candidates on their merits and picks one with no equality assumption. GSM8K is used only because
# its exact-match answers let us benchmark this ranker head-to-head against majority vote and Pass@N.

def _candidate_labels(n: int) -> list[str]:
    """Stable A, B, C, … labels (benchmark widths 1..9 stay within A..I)."""
    return [chr(ord("A") + i) for i in range(n)]


def _build_ranking_block(user_query: str, candidates: list["Candidate"], labels: list[str]) -> str:
    """Question + each candidate's FULL plain-text answer under a letter label. Deliberately does
    NOT inject extracted numbers or vote counts, so the ranker judges the candidates themselves and
    stays independent of majority voting. Showing whole answers (not a normalized key) is also what
    keeps this selector TASK-AGNOSTIC — it works the same for prose, code, or summaries, where no
    comparable answer key exists."""
    parts = [f"Question:\n{user_query}\n"]
    for label, c in zip(labels, candidates):
        parts.append(f"Candidate {label}:\n{c.answer}\n")
    return "\n".join(parts)


def _no_reasoning_prompt(user_query: str, candidates: list["Candidate"], labels: list[str]) -> str:
    """Ranking prompt: choose one label, plain text only, no explanation / no reasoning field."""
    opts = " or ".join(labels)
    return (
        _build_ranking_block(user_query, candidates, labels)
        + "\nDetermine which answer is most likely correct.\n"
        + "Consider:\n"
        + "- arithmetic correctness\n"
        + "- logical consistency\n"
        + "- whether the answer fully addresses the question\n"
        + f"Respond ONLY with a single letter: {opts}.\n"
    )


def _parse_letter_choice(text: Optional[str], labels: list[str]) -> Optional[str]:
    """Map a free-text response to one of `labels`. Tolerates 'B', 'Candidate B', 'B.', 'The answer
    is B', etc. Reading-order scan so the FIRST stated label wins. Returns None if nothing matches
    (caller decides the fallback — never falls back to majority voting)."""
    if not text:
        return None
    label_set = set(labels)
    up = text.strip().upper()
    if up in label_set:                       # exact single-letter reply
        return up
    m = re.search(r"\b([A-Z])\b", up)         # first standalone letter token
    if m and m.group(1) in label_set:
        return m.group(1)
    for ch in up:                             # first valid label char in reading order
        if ch in label_set:
            return ch
    return None


async def rank_no_reasoning(
    user_query: str, candidates: list["Candidate"], llm, idx: str = "?"
) -> "tuple[Candidate, int, bool]":
    """ONE plain-text ranking call over ALL candidates; parse the letter; return the winning
    Candidate. Domain-general: it compares the candidate outputs themselves and needs no answer-equality key, so it applies equally to
    non-numeric tasks (QA, summaries, code, essays) where majority vote is undefined.
    RETURNS: (winner, rank_calls=1, parse_failed). `parse_failed` is True when the reply maps to no
    valid label — a fallback candidate is still returned. Independent of majority."""
    labels = _candidate_labels(len(candidates))
    label_to_cand = dict(zip(labels, candidates))
    prompt = _no_reasoning_prompt(user_query, candidates, labels)

    resp = await llm.ainvoke([HumanMessage(content=prompt)])   # exactly ONE call
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    # TEMPORARY token-usage debug
    _log_token_usage(idx, "Rank Call", msg=resp, prompt_text=prompt, completion_text=text)

    label = _parse_letter_choice(text, labels)
    parse_failed = label is None
    if parse_failed:
        print(f"[IDX {idx}] [rank_no_reasoning] unparseable response {text!r}; falling back to candidate {labels[0]}")
        label = labels[0]
    return label_to_cand[label], 1, parse_failed


# ───────────────────── Self-certainty proxy selector (zero ranking calls) ─────────────────────
# Baseline for the opposite design point: instead of asking the LLM to compare candidates at all,
# score each candidate from the confidence signal already produced while it was generated, then take
# the argmax. Costs ZERO extra LLM calls — that cheapness is the property worth measuring against the
# tournament's (N-1)/(K-1) ranking calls. Requires `generate` to have run with logprobs enabled.

def select_by_self_certainty(candidates: list["Candidate"], idx: str = "?") -> "Candidate":
    """Pick the candidate with the highest self-certainty proxy. NO LLM call.

    RAISES RuntimeError when any candidate lacks a score — that means the server returned no
    logprobs, and silently ranking on partial/absent data would produce numbers that look like a
    valid experiment but measure nothing. Failing loudly here surfaces the misconfiguration on the
    first sample instead of after a full benchmark run."""
    missing = [c.id[:8] for c in candidates if c.confidence is None]
    if missing:
        raise RuntimeError(
            "self_certainty_proxy selected but no logprobs came back for candidate(s) "
            f"{missing}. Check that llama-server supports `logprobs`/`top_logprobs` on "
            "/v1/chat/completions (needs a recent build) and that it is not stripping them."
        )

    ranked = sorted(candidates, key=lambda c: c.confidence, reverse=True)
    for c in ranked:
        print(f"[IDX {idx}] candidate {c.id[:8]}: certainty={c.confidence:.4f}")
    winner = ranked[0]
    print(f"[IDX {idx}] Argmax certainty: {winner.id[:8]} ({winner.confidence:.4f})")
    return winner


# ───────────────────────── Tournament selector (hierarchical bracket ranking) ─────────────────────────
# Pure ORCHESTRATION over the one-shot plain-text ranker (rank_no_reasoning): NO new prompt, NO
# regeneration, NO verifier/critic, NO abstention, NO extra generation round — the generator above is
# untouched and ONLY the selector changes. Rather than one N-way ranking call, run a balanced tournament
# where every ranking call compares at most K candidates (K = `max_group`, default MAX_RANK_GROUP).

def _resolve_max_group(cfg: dict) -> int:
    """Effective tournament group size K for this run: the config value if the harness supplied one
    (--max-group), otherwise MAX_RANK_GROUP.

    Validated ONCE at the top of the selector rather than inside the bracket loop, because the
    failure mode is a hang, not an exception: at K=1 every group is a bye, nobody is eliminated, and
    `tournament_rank` spins forever. A benchmark that hangs on sample 1 of 1319 looks like a slow
    GPU, not a bad flag, so this raises loudly instead."""
    raw = cfg.get("max_rank_group", MAX_RANK_GROUP)
    if raw is None:
        return MAX_RANK_GROUP
    try:
        k = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"max_rank_group must be an integer >= {MIN_RANK_GROUP}, got {raw!r}"
        )
    if k < MIN_RANK_GROUP:
        raise ValueError(
            f"max_rank_group must be >= {MIN_RANK_GROUP}, got {k}. K=1 byes every group, so no "
            "candidate is ever eliminated and the tournament would never terminate."
        )
    return k


def _tournament_groups(
    candidates: list["Candidate"],
    max_group: int = MAX_RANK_GROUP,
) -> list[list["Candidate"]]:
    """One round's bracket split — the MINIMUM-rank-call rule, for any group size K = `max_group`.
    Rank floor(n/K) full groups of K; the trailing n%K candidates each take a BYE (a size-1 group,
    no ranking call) rather than being ranked as a sub-K group. A group of K removes K-1 candidates
    per call, so this hits the theoretical minimum ceil((n-1)/(K-1)) calls for every n and every
    K >= 2 (verified for K=2..5, n=3..16). Byeing the remainder (instead of ranking it) is what lets
    the survivor count drop straight to <=K and finish in one final call — e.g. K=3, N=5: rank one
    [3], bye the other two -> 3 survivors -> 1 final call = 2 calls (NOT [3,2] -> 3 calls).
    Structures at K=3: 3 -> [3] (1 call); 5 -> [3] + 2 byes (2 calls); 7 -> [3,3] + 1 bye (3 calls);
    9 -> [3,3,3] (4 calls). At N=7 the ablation costs 6 calls (K=2), 3 (K=3), 2 (K=4) — so K also
    trades rank compute, not just per-call difficulty. The list is already shuffled by
    `tournament_rank`, so which candidates fill the ranked groups vs. take byes is random; the
    A/B/C… labels are display-only (input order)."""
    n = len(candidates)
    if n <= max_group:
        return [list(candidates)]                         # final group, ranked in one call
    full = (n // max_group) * max_group                    # candidates that fill complete size-K groups
    groups = [list(candidates[i:i + max_group]) for i in range(0, full, max_group)]
    groups.extend([candidates[j]] for j in range(full, n))  # trailing remainder -> singleton byes
    return groups


async def tournament_rank(
    user_query: str,
    candidates: list["Candidate"],
    llm,
    idx: str = "?",
    max_group: int = MAX_RANK_GROUP,
) -> "tuple[Candidate, int, bool, int, int]":
    """Hierarchical tournament selector — an ORCHESTRATION layer over the existing one-shot ranker;
    it does NOT introduce a new ranking prompt. Each ranking call sees at most K = `max_group`
    candidates, decomposing the N-way comparison that degrades at larger N.
    Algorithm: shuffle once -> `_tournament_groups` ranks floor(n/K) groups of K and byes the
    trailing n%K (minimum rank calls) -> rank each group with the one-shot plain-text ranker ->
    advance winners (a byed singleton carries over with no call) -> recurse until one candidate remains.

    RETURNS: (global_winner, total_rank_calls, parse_failed, tournament_rounds,
              tournament_max_group_size).
      - total_rank_calls: real LLM ranking calls only (groups of >= 2); a bye (size 1) costs nothing.
      - parse_failed: True if ANY group call failed to parse a label (a fallback candidate was used).
      - tournament_max_group_size: the largest group actually submitted to a ranking call (0 if none).
    Independent of majority voting; needs no answer-equality key, so it carries over to non-numeric
    tasks exactly like the one-shot ranker it wraps."""
    group_ranker = rank_no_reasoning

    survivors = list(candidates)

    # Global display labels (A, B, C, …) are pinned to the INPUT order — the same order the
    # tournament_select node prints under "Candidates:" — so the bracket log is reconcilable with
    # that listing and with the winner id printed downstream. Assigned BEFORE the shuffle on purpose:
    # the shuffle only randomizes bracket SEEDING (who meets whom), never a candidate's identity
    # label. LOG-ONLY, and independent of the local A/B/C labels each one-shot ranking call assigns
    # internally to its own 2–3 group members. (Pinning to the shuffled order instead made the log
    # impossible to reconcile with the pool listing — a candidate listed first could print as "B".)
    labels = {c.id: lbl for c, lbl in zip(candidates, _candidate_labels(len(candidates)))}

    def lbl(c: "Candidate") -> str:
        return labels.get(c.id, "?")

    random.shuffle(survivors)   # randomize bracket seeding only; identity labels already fixed above

    print(f"\n[IDX {idx}] ================ Tournament Ranking ================\n")
    print(f"[IDX {idx}] Max group size K: {max_group}")

    total_calls = 0
    rounds = 0
    max_group_size = 0
    parse_failed_any = False

    while len(survivors) > 1:
        rounds += 1
        groups = _tournament_groups(survivors, max_group)
        print(f"[IDX {idx}] Round {rounds}")
        next_survivors: list[Candidate] = []
        for gi, group in enumerate(groups, start=1):
            if len(group) == 1:
                # Bye: a lone candidate advances with no ranking call. Under the min-call grouping
                # this is the EXPECTED handling of the trailing n%K (at K=3: N=5 byes 2 -> 3 survivors;
                # N=7 byes 1 -> 3 survivors; N=9 byes none), which is what keeps the call count minimal.
                print(f"[IDX {idx}] Group {gi}: {lbl(group[0])} (bye)")
                print(f"[IDX {idx}] Winner: {lbl(group[0])}\n")
                next_survivors.append(group[0])
                continue

            max_group_size = max(max_group_size, len(group))
            print(f"[IDX {idx}] Group {gi}:")
            if len(group) == 2:
                print(f"[IDX {idx}] {lbl(group[0])} vs {lbl(group[1])}")
            else:                                   # 3-or-more-way group; list vertically
                for c in group:
                    print(f"[IDX {idx}] {lbl(c)}")

            winner, calls, parse_failed = await group_ranker(user_query, group, llm, idx)
            total_calls += calls
            parse_failed_any = parse_failed_any or parse_failed
            print(f"[IDX {idx}] Winner: {lbl(winner)}\n")
            next_survivors.append(winner)
        survivors = next_survivors

    global_winner = survivors[0]

    print(f"[IDX {idx}] Global Winner: {lbl(global_winner)} (id={global_winner.id[:8]})\n")
    print(f"[IDX {idx}] Tournament Rounds: {rounds}")
    print(f"[IDX {idx}] Tournament Rank Calls: {total_calls}")
    print(f"[IDX {idx}] Tournament Max Group Size: {max_group_size}")
    print(f"[IDX {idx}] ====================================================\n")

    return global_winner, total_calls, parse_failed_any, rounds, max_group_size


# ───────────────── Compute-matched USC selector (one-shot ranking, repeated + voted) ─────────────────
# The control that isolates WHY the tournament wins. A raw one-shot ranker spends 1 call while the
# tournament spends (N-1)/(K-1); any accuracy gap could therefore be bought by the extra compute
# rather than earned by the bracket structure. This selector hands the one-shot ranker the SAME call
# budget the tournament would spend, then majority-votes its winners. If the tournament still wins at
# equal budget, the structure is doing the work; if it does not, the gain was compute all along.
#
# Each repeat re-presents the candidates in a fresh random ORDER (rather than re-sampling the same
# prompt). Two reasons: at rank_temperature=0 decoding is deterministic, so identical prompts would
# return identical winners and the vote would be vacuous; and varying order is the strongest use of
# the extra budget for a one-shot ranker, since averaging over orders is exactly what cancels the
# position bias one-shot ranking is known to suffer. It also mirrors `tournament_rank`, which shuffles
# its own bracket seeding — so neither side gets order-luck the other lacks.

def _tournament_call_budget(n: int, max_group: int = MAX_RANK_GROUP) -> int:
    """Ranking calls `tournament_rank` would spend on n candidates at group size K = `max_group`.
    Derived by walking the REAL bracket splitter (`_tournament_groups`) instead of hardcoding
    ceil((n-1)/(K-1)), so the budget cannot drift if K or the bye rule ever changes.

    K MUST be threaded in here, not left at the default: the whole point of this control is to hand
    the one-shot ranker the tournament's exact call budget. At N=7 that budget is 6 calls at K=2 but
    only 2 at K=4, so a K-ablation run that computed the budget at a stale K=3 would silently stop
    being compute-matched and the comparison would measure nothing."""
    calls = 0
    survivors = list(range(n))
    while len(survivors) > 1:
        advancing = []
        for group in _tournament_groups(survivors, max_group):
            if len(group) > 1:          # a bye (size 1) costs no call
                calls += 1
            advancing.append(group[0])
        survivors = advancing
    return calls


async def rank_compute_matched_usc(
    user_query: str, candidates: list["Candidate"], llm, idx: str = "?",
    max_group: int = MAX_RANK_GROUP,
) -> "tuple[Candidate, int, bool, int]":
    """Run the one-shot ranker `_tournament_call_budget(N, K)` times over ALL candidates — each time
    in a freshly shuffled presentation order — then majority-vote the winners.

    Calls run SEQUENTIALLY on purpose: `tournament_rank` awaits its group calls one after another, so
    running these concurrently would hand this baseline a wall-clock advantage that has nothing to do
    with the selection method.

    Ties break toward the candidate earliest in the ORIGINAL pool order — deterministic, so a tie
    never silently injects extra randomness into the result.

    RETURNS: (winner, rank_calls, parse_failed_any, budget)."""
    budget = _tournament_call_budget(len(candidates), max_group)
    pool_index = {c.id: i for i, c in enumerate(candidates)}

    print(
        f"[IDX {idx}] Compute-matched budget: {budget} one-shot call(s) "
        f"for N={len(candidates)}, K={max_group}"
    )

    votes: dict[str, int] = {}
    id_to_cand = {c.id: c for c in candidates}
    parse_failed_any = False

    for r in range(1, budget + 1):
        order = list(candidates)
        random.shuffle(order)           # fresh presentation order -> real variation even at temp 0
        winner, _, parse_failed = await rank_no_reasoning(user_query, order, llm, idx)
        parse_failed_any = parse_failed_any or parse_failed
        votes[winner.id] = votes.get(winner.id, 0) + 1
        print(f"[IDX {idx}] Repeat {r}/{budget}: winner={winner.id[:8]}")

    # Highest vote count; ties -> earliest in the original pool order.
    best_id = min(votes, key=lambda cid: (-votes[cid], pool_index[cid]))
    tally = ", ".join(f"{cid[:8]}={n}" for cid, n in sorted(votes.items(), key=lambda kv: -kv[1]))
    print(f"[IDX {idx}] Vote tally: {tally}")
    print(f"[IDX {idx}] Vote winner: {best_id[:8]} ({votes[best_id]}/{budget})")

    return id_to_cand[best_id], budget, parse_failed_any, budget


async def tournament_select(state: AgentState, config: RunnableConfig) -> dict:
    """Tournament selector node. Generation already produced `pool`; here we select Top-1 via a
    HIERARCHICAL TOURNAMENT (`tournament_rank`) instead of one N-way ranking call, to avoid the
    degradation one-shot ranking shows as candidate count grows. Pure orchestration over the
    one-shot plain-text ranker; the generator is untouched.
    Width=1 short-circuits (one candidate, no call, 0 rounds). Preserves EVERY majority/one-shot metric
    and adds tournament_rounds / tournament_rank_calls / tournament_max_group_size. `rank_calls`
    equals `tournament_rank_calls` (both count the tournament's ranking LLM calls), so the existing
    eval aggregation keeps working unchanged.
    RETURNS: final_answer, messages, the selector-log fields, and the three tournament metrics."""
    cfg = config["configurable"]
    pool = state.get("pool", [])
    user_query = state.get("user_query", "")
    idx = _sample_idx(state, config)
    max_group = _resolve_max_group(cfg)   # K for this run; raises on K < 2 rather than hanging

    sampled_answers = [
        c.answer
        for c in pool
    ]

    print(f"\n[IDX {idx}] ============ Tournament-Select Node ============\n")
    print(f"[IDX {idx}] Rank mode: Tournament")
    print(f"[IDX {idx}] Max group size K: {max_group}")
    print(f"[IDX {idx}] Candidates: {len(pool)}")
    for label, candidate in zip(_candidate_labels(len(pool)), pool):
        print(f"[IDX {idx}] {label}: id={candidate.id[:8]}")

    # sampled_answers, sampled_numbers, vote_distribution, unique_answers = _self_consistency_metrics(pool)

    if not pool:
        print(f"[IDX {idx}] Empty pool.")
        print(f"\n[IDX {idx}] ================================================================\n")
        return {
            "final_answer": "",
            "messages": [AIMessage(content="")],
            "sampled_answers": [],
            "rank_latency": 0.0,
            "rank_calls": 0,
            "winner_candidate_id": None,
            "rank_parse_failed": False,
            "tournament_rounds": 0,
            "tournament_rank_calls": 0,
            "tournament_max_group_size": 0,
        }

    rank_latency = 0.0
    rank_calls = 0
    rank_parse_failed = False
    tournament_rounds = 0
    tournament_max_group_size = 0

    if len(pool) <= 1:
        # Single candidate: nothing to select (no LLM call, no bracket).
        winner = pool[0]
        print(f"[IDX {idx}] Single candidate; no tournament needed.")
    elif cfg.get("rank_mode") == "self_certainty_proxy":
        print(f"[IDX {idx}] Selecting by self-certainty proxy over {len(pool)} candidates (0 LLM calls)...")
        t0 = time.perf_counter()
        winner = select_by_self_certainty(pool, idx)
        rank_latency = time.perf_counter() - t0     # local arithmetic only; expected to be ~0
        rank_calls = 0                              # the whole point: no ranking call at all
        tournament_rounds = 0
        tournament_max_group_size = 0
    elif cfg.get("rank_mode") == "compute_matched_usc":
        rank_temp = cfg.get("rank_temperature", 0.0)
        llm = get_llm(config, temperature=rank_temp)
        print(f"[IDX {idx}] Executing Compute-Matched USC over all {len(pool)} candidates...")
        if rank_temp == 0.0:
            print(f"[IDX {idx}] (rank_temperature=0 -> repeats differ by candidate ORDER only)")
        t0 = time.perf_counter()
        winner, rank_calls, rank_parse_failed, _budget = await rank_compute_matched_usc(
            user_query, pool, llm, idx, max_group
        )
        rank_latency = time.perf_counter() - t0
        # Single-stage selector like rank_no_reasoning: every call sees all N candidates, so reuse the
        # same metric semantics (rounds=1, max group = full pool) instead of inventing new fields.
        tournament_rounds = 1
        tournament_max_group_size = len(pool)
    elif cfg.get("rank_mode") in ("pairjudge", "rrm"):
        # External reward-model judge (PairJudge RM / RRM). These are HuggingFace checkpoints, not the
        # llama-server client, so this path never touches get_llm(). Imported here rather than at
        # module scope so runs that don't use them need no torch/transformers install.
        from rm_judges import rm_knockout
        kind = cfg["rank_mode"]
        # "numeric" reproduces PairJudge's answer-equality teams (valid on GSM8K); "none" gives every
        # candidate its own team, which is what HumanEval requires — see rm_judges' module docstring.
        grouping = cfg.get("rm_team_grouping", "none")
        print(f"[IDX {idx}] Executing {kind} knockout over {len(pool)} candidates...")
        t0 = time.perf_counter()
        winner, rank_calls, rank_parse_failed, tournament_rounds = await rm_knockout(
            user_query, pool, kind, grouping, idx, sample_tag=str(cfg.get("sample_idx", "")),
        )
        rank_latency = time.perf_counter() - t0
        # Pairwise by construction for both judges (their published algorithms are K=2 knockouts),
        # so --max-group does NOT apply here and is deliberately not passed through.
        tournament_max_group_size = 2
    elif cfg.get("rank_mode") == "rank_no_reasoning":
        print(f"[IDX {idx}] Executing One-Shot Ranker over all {len(pool)} candidates...")
        rank_temp = cfg.get("rank_temperature", 0.0)
        llm = get_llm(config, temperature=rank_temp)
        t0 = time.perf_counter()
        winner, rank_calls, rank_parse_failed = await rank_no_reasoning(
            user_query, pool, llm, idx
        )
        rank_latency = time.perf_counter() - t0
        tournament_rounds = 1
        tournament_max_group_size = len(pool)
    else:
        rank_temp = cfg.get("rank_temperature", 0.0)   # greedy ranker by default -> reproducible
        llm = get_llm(config, temperature=rank_temp)
        t0 = time.perf_counter()
        (winner, rank_calls, rank_parse_failed,
         tournament_rounds, tournament_max_group_size) = await tournament_rank(
            user_query, pool, llm, idx, max_group
        )
        rank_latency = time.perf_counter() - t0

    answer = winner.answer

    print(f"[IDX {idx}] Winner: id={winner.id[:8]}")
    print(f"[IDX {idx}] Tournament rounds: {tournament_rounds} | rank calls: {rank_calls} | max group: {tournament_max_group_size}")
    print(f"[IDX {idx}] Rank latency: {rank_latency:.3f}s")
    print(f"[IDX {idx}] Parse failed: {rank_parse_failed}")
    print(f"\n[IDX {idx}] ================================================================\n")

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "sampled_answers": sampled_answers,
        "rank_latency": rank_latency,
        "rank_calls": rank_calls,
        "winner_candidate_id": winner.id,
        "rank_parse_failed": rank_parse_failed,
        "tournament_rounds": tournament_rounds,
        "tournament_rank_calls": rank_calls,
        "tournament_max_group_size": tournament_max_group_size,
    }


# ─── Build graph (setup -> generate -> tournament_select) ───

builder = StateGraph(AgentState)
builder.add_node("setup", setup)
builder.add_node("generate", generate)
builder.add_node("tournament_select", tournament_select)  # hierarchical tournament selector

builder.add_edge(START, "setup")
builder.add_edge("setup", "generate")
builder.add_edge('generate', 'tournament_select')
builder.add_edge("tournament_select", END)

GRAPH_RANKER = builder.compile()


# ───────────────────────── Local smoke test ─────────────────────────

def _fresh_state(question: str) -> AgentState:
    return {
        "messages": [{"role": "user", "content": question}],
        "user_query": "",
        "reasoning_mode": "low",

        "pool": [],

        "width": 1,
        "final_answer": None,
        "sampled_answers": [],

        "rank_latency": 0.0,
        "rank_calls": 0,
        "winner_candidate_id": None,
        "rank_parse_failed": False,

        "tournament_rounds": 0,
        "tournament_rank_calls": 0,
        "tournament_max_group_size": 0,
    }


if __name__ == "__main__":
    config = {
        "configurable": {
            "model_name": "qwen",
            "reasoning_mode": "low",        # low=1; medium=3; high=5; max=7
            "temperature": 0.0,             # low: 0.0 reproduces baseline. medium/high: use >0 (e.g. 0.7)
            "rank_temperature": 0.0,            # ranker decoding temp (greedy by default)
        }
    }
    result = asyncio.run(GRAPH_RANKER.ainvoke(_fresh_state("What is 17 * 24?"), config=config))
    print(result["final_answer"])