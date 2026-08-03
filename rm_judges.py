"""External reward-model judges (PairJudge RM, RRM) used as ranking backends.

These are the two published knockout-tournament Best-of-N methods this project compares against.
Both are ordinary HuggingFace checkpoints driven with `transformers`, NOT the OpenAI-compatible
llama-server the generator uses — so they cannot go through `get_llm()` and live here instead.

FIDELITY: every prompt, decoding setting, parser and bracket rule below is ported from the authors'
own released code so the comparison measures THEIR method, not our reinterpretation of it:
  - PairJudge RM prompt   : prompts/compare_0_ex.md   (github.com/THU-KEG/PairwiseRM), verbatim
  - PairJudge RM protocol : pairwise/compare_resp.py  — system="You are a helpful assistant.",
                            max_tokens=min(2048, 4096-prompt-200), temperature=0, top_p=1,
                            stop="</resp_b_judge>"
  - PairJudge RM parser   : pairwise/compare_resp.py::extract_judge_result, ported below
  - PairJudge RM bracket  : pairwise/knockout.py::get_champion, ported below (teams, byes,
                            same-team shortcut, early termination, BOTH-eliminated rule)
  - RRM prompt + decoding : the Reward-Reasoning/RRM-7B model card, verbatim
                            (max_new_tokens=8192, temperature=0.6, do_sample=True, top_p=1.0)

TEAM GROUPING is the one place the two datasets differ, and it is the crux of the experiment.
PairJudge groups candidates into TEAMS by identical extracted answer; same-team members never fight
(saving calls) and the whole bracket ends early once one team is all that is left. That grouping is
well defined on GSM8K (`grouping="numeric"`) and UNDEFINED on HumanEval, where two correct programs
almost never match as strings. For HumanEval we therefore pass `grouping="none"` (every candidate is
its own team), which isolates the domain-general half of their method — pairwise CoT judging — from
the domain-specific half that does not transfer. Report that choice explicitly; it is a deliberate
adaptation, not their algorithm run unchanged.

Set RM_JUDGMENT_LOG=<path.jsonl> to record every pairwise judgment (both answers, the verdict, the
full chain-of-thought, and cheap keyword flags) for the pre-registered failure analysis.
"""

import os
import re
import json
import random
import asyncio
import threading
from typing import Optional

# `transformers`/`torch` are imported lazily inside _load so that runs which never select an RM
# rank_mode do not need them installed at all.

PAIRJUDGE_MODEL_ID = os.getenv("PAIRJUDGE_MODEL_ID", "THU-KEG/PairJudge-RM")
RRM_MODEL_ID = os.getenv("RRM_MODEL_ID", "Reward-Reasoning/RRM-7B")

# PairJudge: the authors cap generation at 2048 and further shrink it to keep prompt+output inside a
# 4096 window (compare_resp.py). Both numbers are theirs.
PAIRJUDGE_MAX_NEW_TOKENS = int(os.getenv("PAIRJUDGE_MAX_NEW_TOKENS", "2048"))
PAIRJUDGE_CTX_BUDGET = int(os.getenv("PAIRJUDGE_CTX_BUDGET", "4096"))

# RRM: kept at the model card's 8192 on purpose. Its verdict comes AFTER a long chain of thought, so
# truncating risks cutting off the \boxed{...} and turning a real judgment into a parse failure —
# that would look like a method weakness when it is only a budget artifact.
RRM_MAX_NEW_TOKENS = int(os.getenv("RRM_MAX_NEW_TOKENS", "8192"))
RRM_TEMPERATURE = float(os.getenv("RRM_TEMPERATURE", "0.6"))
RRM_TOP_P = float(os.getenv("RRM_TOP_P", "1.0"))

JUDGMENT_LOG_PATH = os.getenv("RM_JUDGMENT_LOG")

# ───────────────────────── Prompts (verbatim from the authors) ─────────────────────────

# prompts/compare_0_ex.md, unmodified. Note what it asks for: a "Math Judgement Task", step-by-step
# verification of "Mathematical accuracy", "algebraic simplifications", re-deriving calculations. On
# HumanEval this template is applied to Python functions — the mismatch is the thing under test, so
# it is deliberately NOT rewritten for code.
PAIRJUDGE_TEMPLATE = """### **Math Judgement Task for AI Assistant Responses**

**Task Objective:**  
Evaluate the correctness of two responses (Response A and Response B) to a given math question. Perform a step-by-step verification of each response's accuracy. After completing the step-by-step checks, provide a final correctness judgment for each response.

### **Steps to Follow:**

0. **Extract Answers from both Responses:**
   - Read and both responses to identify the final answers provided.
   - If both responses provide the same answer, make sure that the final judgment would be the same for both responses, either both correct or both incorrect.
   - If the responses provide different answers, make sure there are is no possible way that both responses can be correct. It must be the case that one response is correct and the other is incorrect or both are incorrect.


1. **Step-by-Step Verification of Correctness:**
   - **For each response (Response A and Response B):**  
     Carefully examine each step of the solution provided. Check the following:
     - **Mathematical accuracy:** Ensure all calculations, algebraic simplifications, and mathematical operations are correct.
     - **Logical consistency:** Verify that each step follows logically from the previous one and that the reasoning is sound.
     - **Completeness:** Make sure that all necessary steps are included to fully solve the problem and reach the final answer.

   While performing this step-by-step evaluation, refer to the **Additional Tips** section for helpful techniques to validate each response's accuracy.  
   **Attention:** When checking the correctness of a single step, you should never first conclude the correctness of this step (for example, *"This step is incorrect because..."* is strictly forbidden). You should neutrally check this step, provide evidence about its correctness, and then finally draw a conclusion about the correctness of this step. In other words, you should first employ the techniques in **Additional Tips** to check the correctness of this step, and then draw a conclusion about the correctness of this step.

2. **Final Conclusion:**
   - After completing the step-by-step verification for each response, sum up the information you have now, then finally determine whether each response's answer is **correct** or **incorrect**.
   - Provide the final judgment for each response, the output should in-closed with the following tags:
     - If Response A's answer is correct:  
       <resp_a_judge>Correct</resp_a_judge>
     - If Response A's answer is incorrect:  
       <resp_a_judge>Incorrect</resp_a_judge>
     - If Response B's answer is correct:  
       <resp_b_judge>Correct</resp_b_judge>
     - If Response B's answer is incorrect:  
       <resp_b_judge>Incorrect</resp_b_judge>
   - **Note:** The responses A and response B can be either correct or incorrect, or both correct, or both incorrect. You should provide the final judgment for each response. There is no guarantee that at least one response is correct or incorrect.
### **Additional Tips:**

- **Key Validation Techniques (to apply during Step 1):**
  - **Re-derive Key Parts of the Solution:** Independently calculate or derive crucial steps of the solution to verify their correctness.
  - **Verify Calculations:** Double-check all mathematical operations (e.g., addition, multiplication, division) to confirm accuracy.
  - **Compare Responses:** If needed, compare similar steps between Response A's and Response B's answers to identify discrepancies or inconsistencies.

- **The final output format** should be as follows:
### **Final Judgment:**
**Response A:** <resp_a_judge>Correct/Incorrect</resp_a_judge>
**Response B:** <resp_b_judge>Correct/Incorrect</resp_b_judge>

ALWAYS REMEMBER: inclosed the final judgment with the corresponding tags <resp_a_judge> and <resp_b_judge> to indicate the correctness of each response.

### **Question:**
<question>
{question}
</question>

### **Response A:**
<response_a>
{response_a}
</response_a>

### **Response B:**
<response_b>
{response_b}
</response_b>
"""

PAIRJUDGE_SYSTEM = "You are a helpful assistant."
PAIRJUDGE_STOP = "</resp_b_judge>"

# Reward-Reasoning/RRM-7B model card, unmodified. Unlike PairJudge this asks for a RELATIVE
# preference ("select the best response"), never absolute correctness — which is why RRM has no
# both-eliminated outcome and needs no team grouping.
RRM_SYSTEM_PROMPT = """You are a helpful assistant in evaluating the quality of the responses for a given instruction. Your goal is to select the best response for the given instruction.
Select Assistant 1 or Assistant 2, that is better for the given instruction. The two responses are generated by two different AI assistants respectively.
Do NOT say both / neither are good.

Here are some rules of the evaluation:
(1) If the instruction does not contain harmful content, you should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) If the instruction contains harmful content, prioritize the harmlessness and safety of the response.
(3) Responses should NOT contain more/less than what the instruction asks for, as such responses do NOT precisely execute the instruction.
(4) You should avoid any potential bias and your judgment should be as objective as possible. Here are some potential sources of bias:
- The order in which the responses were presented should NOT affect your judgment, as Response A and Response B are equally likely to be the better.
- The length of the responses should NOT affect your judgement, as a longer response does not necessarily correspond to a better response. When making your decision, evaluate if the response length is appropriate for the given instruction.
(5) Your output should only consist of '\\boxed{Assistant 1}' if assistant 1 is better, or '\\boxed{Assistant 2}' if assistant 2 is better. Omit any other output.

"""

RRM_PROMPT = """## Query

{question}

## Assistant responses

### Assistant 1

{answer1}


### Assistant 2

{answer2}

"""

RRM_ASSISTANT_PROMPT = """## Analysis

Let's analyze this step by step and decide which assistant is better, and then answer \\boxed{Assistant 1} or \\boxed{Assistant 2}."""


# ───────────────────────── Model loading (lazy, one at a time on the GPU) ─────────────────────────

_MODELS: dict = {}
_LOAD_LOCK = threading.Lock()
# Serializes .generate() across threads. The eval harness runs many dataset items concurrently, but
# they all share ONE GPU: letting several 7-8B generations interleave there trades throughput for
# memory pressure without any gain, so judgments are executed strictly one at a time.
_GPU_LOCK = threading.Lock()


def _model_id(kind: str) -> str:
    return PAIRJUDGE_MODEL_ID if kind == "pairjudge" else RRM_MODEL_ID


def _load(kind: str):
    """Load (once) the tokenizer+model for `kind`. Only the requested judge is ever loaded, so a
    PairJudge run never pays for RRM's weights or vice versa."""
    if kind in _MODELS:
        return _MODELS[kind]
    with _LOAD_LOCK:
        if kind in _MODELS:                       # another thread finished while we waited
            return _MODELS[kind]
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = _model_id(kind)
        print(f"[rm_judges] Loading {kind} judge: {model_id} (first use; this takes a while)")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=os.getenv("RM_DEVICE_MAP", "auto"),
        )
        model.eval()
        _MODELS[kind] = (tok, model)
        print(f"[rm_judges] {kind} judge ready.")
        return _MODELS[kind]


def _generate(kind: str, messages: list, max_new_tokens: int, **gen_kwargs) -> str:
    """Run one judge generation and return ONLY the newly generated text (prompt echo stripped)."""
    import torch

    tok, model = _load(kind)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with _GPU_LOCK:
        with torch.no_grad():
            try:
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                    **gen_kwargs,
                )
            except TypeError as e:
                # `stop_strings` needs transformers >= 4.39. Older versions raise here; drop the stop
                # sequence and let generation run to EOS/limit instead of crashing mid-benchmark. The
                # parser handles the untruncated text fine — it only costs extra tokens.
                if "stop_strings" not in gen_kwargs:
                    raise
                print(f"[rm_judges] NOTE: stop_strings unsupported by this transformers build "
                      f"({e}); retrying without it.")
                gen_kwargs.pop("stop_strings", None)
                gen_kwargs.pop("tokenizer", None)
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                    **gen_kwargs,
                )
    return tok.decode(out[0][input_len:], skip_special_tokens=True)


# ───────────────────── PairJudge RM: verdict parsing (ported from the authors) ─────────────────────

def _convert_str_to_bool(str_val: Optional[str]) -> Optional[bool]:
    """compare_resp.py::convert_str_to_bool — 'true'/'correct' -> True, 'false'/'incorrect' -> False,
    anything else -> None (unparseable)."""
    if str_val is None:
        return None
    v = str_val.strip().lower()
    if v in ("true", "correct"):
        return True
    if v in ("false", "incorrect"):
        return False
    return None


def extract_judge_result(judge_content: str) -> "tuple[Optional[bool], Optional[bool]]":
    """compare_resp.py::extract_judge_result — port of the authors' tolerant parser, including its
    fallbacks for models that answer with \\box/\\boxed/':**True' instead of the required tags.
    RETURNS (response_A_correct, response_B_correct); None means the verdict could not be read."""
    if not judge_content:
        return None, None
    t = judge_content.replace(" ", "").lower()

    a = b = None
    if "<resp_a_judge>" in t and "<resp_b_judge>" in t:
        ma = re.search(r"<resp_a_judge>(.*?)</resp_a_judge>", t, re.DOTALL)
        mb = re.search(r"<resp_b_judge>(.*?)</resp_b_judge>", t, re.DOTALL)
        a = _convert_str_to_bool(ma.group(1) if ma else None)
        b = _convert_str_to_bool(mb.group(1) if mb else None)
        return a, b

    if "\\box{true}" in t or "\\box{false}" in t:
        m = re.findall(r"\\box\{(true|false)\}", t)
        if len(m) >= 2:
            return _convert_str_to_bool(m[0]), _convert_str_to_bool(m[1])
    elif "\\boxed{true}" in t or "\\boxed{false}" in t:
        m = re.findall(r"\\boxed\{(true|false)\}", t)
        if len(m) >= 2:
            return _convert_str_to_bool(m[0]), _convert_str_to_bool(m[1])
    elif "\\boxed{\\text{true}}" in t or "\\boxed{\\text{false}}" in t:
        m = re.findall(r"\\boxed\{\\text\{(true|false)\}\}", t, re.IGNORECASE)
        if len(m) >= 2:
            return _convert_str_to_bool(m[0]), _convert_str_to_bool(m[1])
        if len(m) == 1:
            return _convert_str_to_bool(m[0]), None
        return None, None
    elif ":**true" in t or ":**false" in t:
        m = re.findall(r":\*\*(true|false)", t, re.IGNORECASE)
        if len(m) >= 2:
            return _convert_str_to_bool(m[0]), _convert_str_to_bool(m[1])
    elif "responsea:true" in t or "responsea:false" in t:
        return ("responsea:true" in t), ("responseb:true" in t)
    elif "bothresponsesarecorrect" in t:
        return True, True
    elif "bothresponsesareincorrect" in t:
        return False, False

    return a, b


def _judge_pairjudge(question: str, resp_a: str, resp_b: str) -> "tuple[Optional[bool], Optional[bool], str]":
    """One PairJudge RM comparison. RETURNS (a_correct, b_correct, raw_cot)."""
    prompt = PAIRJUDGE_TEMPLATE.format(question=question, response_a=resp_a, response_b=resp_b)

    # Authors' budget rule: keep prompt+completion inside PAIRJUDGE_CTX_BUDGET, never above 2048.
    tok, _ = _load("pairjudge")
    used = len(tok(prompt)["input_ids"])
    max_new = min(PAIRJUDGE_MAX_NEW_TOKENS, PAIRJUDGE_CTX_BUDGET - used - 200)
    if max_new <= 0:
        max_new = PAIRJUDGE_MAX_NEW_TOKENS

    text = _generate(
        "pairjudge",
        [{"role": "system", "content": PAIRJUDGE_SYSTEM},
         {"role": "user", "content": prompt}],
        max_new_tokens=max_new,
        do_sample=False,                 # temperature=0 in the reference implementation
        stop_strings=[PAIRJUDGE_STOP],
        tokenizer=tok,                   # required by transformers when using stop_strings
    )
    # The reference client strips the stop sequence and adds it back before parsing; HF usually keeps
    # it, so only append when it is genuinely missing (the parser needs the closing tag).
    if PAIRJUDGE_STOP not in text and "<resp_b_judge>" in text:
        text += PAIRJUDGE_STOP

    a, b = extract_judge_result(text)
    return a, b, text


# ───────────────────────── RRM: verdict parsing ─────────────────────────

_RRM_CHOICE_RE = re.compile(r"\\boxed\{\s*assistant\s*([12])\s*\}", re.IGNORECASE)


def _judge_rrm(question: str, resp_a: str, resp_b: str) -> "tuple[Optional[str], str]":
    """One RRM comparison. RETURNS ('A' | 'B' | None, raw_cot); None means unparseable.
    Assistant 1 == response A, Assistant 2 == response B."""
    user = RRM_PROMPT.format(question=question, answer1=resp_a, answer2=resp_b) + RRM_ASSISTANT_PROMPT
    text = _generate(
        "rrm",
        [{"role": "system", "content": RRM_SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        max_new_tokens=RRM_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=RRM_TEMPERATURE,
        top_p=RRM_TOP_P,
    )
    matches = _RRM_CHOICE_RE.findall(text)
    if not matches:
        return None, text
    return ("A" if matches[-1] == "1" else "B"), text   # last box wins: it is the final verdict


# ───────────────────── Judgment logging + cheap failure-mode flags ─────────────────────

# Pre-registered Tier-3 signals: does the judge reason about CODE, or is it hunting for a numeric
# answer the way its math training expects? These are coarse keyword counts written at log time so
# the classification is fixed BEFORE any transcript is read, not fitted to what the outputs happen to
# say. They are evidence to inspect, never the headline metric — that stays pairwise accuracy against
# unit-test ground truth, computed by the harness.
_MATH_LANG_RE = re.compile(
    r"\b(calculat\w*|arithmetic|multiplicat\w*|multiply|addition|subtract\w*|divis\w*|"
    r"algebra\w*|equation|re-derive|final answer|correct value|numerical)\b", re.IGNORECASE)
_CODE_LANG_RE = re.compile(
    r"\b(code|function|syntax|variable|loop|return|edge case|test case|implementation|"
    r"algorithm|python|docstring|parameter)\b", re.IGNORECASE)

_LOG_LOCK = threading.Lock()


def _log_judgment(record: dict) -> None:
    """Append one judgment to RM_JUDGMENT_LOG (JSONL). No-op when the env var is unset."""
    if not JUDGMENT_LOG_PATH:
        return
    record["math_lang_hits"] = len(_MATH_LANG_RE.findall(record.get("raw_cot") or ""))
    record["code_lang_hits"] = len(_CODE_LANG_RE.findall(record.get("raw_cot") or ""))
    try:
        with _LOG_LOCK:
            with open(JUDGMENT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:                      # logging must never break a benchmark run
        print(f"[rm_judges] WARNING: could not write judgment log: {e}")


# ───────────────────────── Team grouping ─────────────────────────

_NUM = r"[-+]?\d[\d,]*\.?\d*"


def _extract_final_number(text: str) -> Optional[float]:
    """Final numeric answer, same precedence as the GSM8K harness (#### -> 'answer is' -> \\boxed ->
    last number). Duplicated rather than imported to keep this module free of harness imports; keep
    the two in sync if the harness rule ever changes."""
    if not text:
        return None
    text = text.strip()
    for pat in (rf"####\s*({_NUM})",
                rf"(?:the\s+)?answer\s+is\s+\$?\s*({_NUM})",
                rf"\\boxed\{{({_NUM})\}}"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", ""))
    nums = re.findall(rf"(?<!\w)({_NUM})(?!\w)", text)
    return float(nums[-1].replace(",", "")) if nums else None


def _team_ids(candidates: list, grouping: str) -> list:
    """Team label per candidate.
    grouping="numeric": PairJudge's real rule — candidates sharing an extracted final answer form one
      team. Well defined on GSM8K. Candidates with no extractable number get a singleton team.
    grouping="none":    every candidate is its own team. Used on HumanEval, where no answer-equality
      key exists (see module docstring)."""
    if grouping != "numeric":
        return [c.id for c in candidates]
    ids = []
    for c in candidates:
        n = _extract_final_number(c.answer)
        ids.append(f"num:{n}" if n is not None else f"uniq:{c.id}")
    return ids


# ───────────────────────── Knockout bracket (ported from knockout.py) ─────────────────────────

async def _match_pair(kind, question, item_a, item_b, idx, sample_tag):
    """One bracket match. RETURNS (winners, llm_calls, parse_failed).

    `winners` may hold 0, 1 or 2 entries, matching the reference implementation:
      - same team (PairJudge only)  -> a coin flip, NO model call (their `match_pair` shortcut)
      - PairJudge, A correct        -> [A];  B correct -> [B]
      - PairJudge, neither correct  -> []  BOTH ELIMINATED. This is their rule, not a bug: the judge
        rules on absolute correctness, so a match can knock out both entrants. It is also why the
        bracket can empty out (handled by the caller).
      - RRM                         -> always exactly one winner (it is a relative preference model
        and is instructed never to answer "neither")."""
    team_a, cand_a = item_a
    team_b, cand_b = item_b

    if team_a == team_b:
        # Same answer -> nothing to decide, and the authors deliberately spend no call here.
        return [random.choice([item_a, item_b])], 0, False

    if kind == "pairjudge":
        a_ok, b_ok, raw = await asyncio.to_thread(_judge_pairjudge, question, cand_a.answer, cand_b.answer)
        parse_failed = a_ok is None or b_ok is None
        if a_ok:
            winners, verdict = [item_a], "A"
        elif b_ok:
            winners, verdict = [item_b], "B"
        else:
            winners, verdict = [], ("PARSE_FAIL" if parse_failed else "NEITHER")
    else:
        choice, raw = await asyncio.to_thread(_judge_rrm, question, cand_a.answer, cand_b.answer)
        parse_failed = choice is None
        if parse_failed:
            # Forced-choice model: falling back to A keeps the bracket alive; flagged and logged so
            # these matches can be excluded from analysis if they turn out to be common.
            winners, verdict = [item_a], "PARSE_FAIL"
        else:
            winners, verdict = ([item_a] if choice == "A" else [item_b]), choice

    print(f"[IDX {idx}] {cand_a.id[:8]} vs {cand_b.id[:8]} -> {verdict}")

    _log_judgment({
        "sample": sample_tag, "idx": idx, "judge": kind, "question": question,
        "cand_a_id": cand_a.id, "cand_b_id": cand_b.id,
        "answer_a": cand_a.answer, "answer_b": cand_b.answer,
        "verdict": verdict, "parse_failed": parse_failed, "raw_cot": raw,
    })
    return winners, 1, parse_failed


async def rm_knockout(question, candidates: list, kind: str, grouping: str = "none",
                      idx: str = "?", sample_tag: str = "") -> "tuple[object, int, bool, int]":
    """Knockout tournament judged by an external reward model.

    Port of pairwise/knockout.py::get_champion, with one deliberate change: matches run SEQUENTIALLY
    instead of in a ThreadPoolExecutor, because all judgments share one GPU (see _GPU_LOCK).

    RETURNS (winner_candidate, llm_calls, parse_failed_any, rounds)."""
    if not candidates:
        raise ValueError("rm_knockout called with an empty candidate pool")
    if len(candidates) == 1:
        return candidates[0], 0, False, 0

    if kind == "rrm" and grouping == "numeric":
        # RRM has no notion of teams — it is a pairwise preference model, so answer-equality grouping
        # would be an invention of ours, not part of their method.
        print(f"[IDX {idx}] (rrm ignores team grouping; every candidate is its own team)")
        grouping = "none"

    teams = _team_ids(candidates, grouping)
    items = list(zip(teams, candidates))
    distinct = len(set(teams))

    print(f"\n[IDX {idx}] ============ {kind} knockout ({len(candidates)} candidates, "
          f"{distinct} team(s), grouping={grouping}) ============")

    calls = 0
    rounds = 0
    parse_failed_any = False

    while len(items) > 1:
        if len({t for t, _ in items}) == 1:
            # Everything left agrees; the reference implementation stops here rather than paying for
            # matches whose outcome cannot change the answer.
            print(f"[IDX {idx}] Early termination: all survivors share one team")
            break

        rounds += 1
        round_start = list(items)
        next_round = []

        if len(items) % 2 == 1:                    # odd count -> one random bye, as in the original
            bye_i = random.randrange(len(items))
            bye = items.pop(bye_i)
            next_round.append(bye)
            print(f"[IDX {idx}] Round {rounds}: {bye[1].id[:8]} gets a bye")

        available = items[:]
        random.shuffle(available)
        pairs = []
        while len(available) >= 2:
            a = available.pop()
            b = None
            for j, cand in enumerate(available):
                if cand[0] != a[0]:                # prefer an opponent from a different team
                    b = available.pop(j)
                    break
            pairs.append((a, b if b is not None else available.pop()))

        for a, b in pairs:
            winners, c, pf = await _match_pair(kind, question, a, b, idx, sample_tag)
            calls += c
            parse_failed_any = parse_failed_any or pf
            next_round.extend(winners)

        # Every match can eliminate both entrants, so a round may wipe the bracket out. The reference
        # implementation falls back to a random survivor from the round that just ran.
        items = next_round if next_round else [random.choice(round_start)]

    winner = items[0][1]
    print(f"[IDX {idx}] {kind} winner: {winner.id[:8]} | rounds={rounds} | calls={calls}\n")
    return winner, calls, parse_failed_any, rounds
