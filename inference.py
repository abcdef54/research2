import asyncio
import re
import os
import csv
import argparse
import math
import time
from tqdm import tqdm
from langchain_core.messages import HumanMessage

# HumanEval specific primitives
from human_eval.data import read_problems
from human_eval.execution import check_correctness

# Your LangGraph tournament orchestration system
from final_ranker import GRAPH_RANKER, AgentState

os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Global semaphore configured via args
SEM = None

def extract_code_completion(text: str, prompt: str = "", entry_point: str = "") -> str:
    """Extract code continuation from the model response.
    
    1. Removes markdown code fences if present using regex r"```(?:python)?\s*(.*?)```".
    2. If no markdown fences are found, leaves the text unchanged.
    3. If the model repeated the prompt or the function signature/docstring, strips it 
       to keep only the continuation.
    4. Indentation is strictly preserved.
    """
    if not text:
        return ""

    # Improve extract_code_completion() using a regex that strips markdown fences if present
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        content = match.group(1)
    else:
        content = text

    # Remove duplicated prompt or function signature/docstring if the model repeated it.
    sig_match = None
    if entry_point:
        # Search for the function signature of the entry point
        # e.g., def entry_point(...) -> ret:
        # We use re.DOTALL in case the signature spans multiple lines
        sig_match = re.search(rf"def\s+{entry_point}\s*\(.*?\)\s*(?:->\s*.*?)?:", content, re.DOTALL)
        if sig_match:
            end_sig = sig_match.end()
            pre_sig = content[:sig_match.start()]
            
            # Extract any lines starting with import or from (ignoring leading whitespace)
            imports = []
            for line in pre_sig.splitlines():
                stripped_line = line.strip()
                if stripped_line.startswith("import ") or stripped_line.startswith("from "):
                    imports.append(stripped_line)
            
            remaining = content[end_sig:]
            # Check if there is an optional docstring following the signature
            # We match triple or sextuple quotes, followed by content, closing quotes, and optional trailing spaces/newline.
            doc_match = re.match(r'^\s*("""""\"|\'\'\'\'\'\'|\"\"\"|\'\'\').*?\1[ \t]*\r?\n?', remaining, re.DOTALL)
            if doc_match:
                body = remaining[doc_match.end():]
            else:
                body = remaining
                
            # Find the indentation of the first non-empty line of body
            indent = ""
            for line in body.splitlines():
                if line.strip():
                    indent = line[:len(line) - len(line.lstrip())]
                    break
            
            # Indent each import line to match the body indentation
            indented_imports = []
            for imp in imports:
                indented_imports.append(indent + imp)
                
            if indented_imports:
                content = "\n".join(indented_imports) + "\n" + body
            else:
                content = body

    # Fallback prompt duplication detection (if entry_point wasn't provided or signature wasn't found)
    if prompt and (not entry_point or not sig_match):
        # Normalize newlines to avoid platform-specific discrepancies
        p_norm = prompt.replace("\r\n", "\n")
        c_norm = content.replace("\r\n", "\n")
        pos = c_norm.find(p_norm)
        # If the prompt is found right at the beginning (ignoring any leading whitespaces/newlines)
        if pos != -1 and c_norm[:pos].strip() == "":
            content = c_norm[pos + len(p_norm):]

    return content


async def solve(
    question: str,
    model_name: str,
    reasoning_mode: str,
    temperature: float,
    rank_mode: str,
    rank_temperature: float
):
    config = {
        "configurable": {
            "model_name": model_name,
            "personality": "general",
            "tools_enabled": False,
            "reasoning_mode": reasoning_mode,
            "bypass_governor": True,
            "temperature": temperature,
            "rank_mode": rank_mode,
            "rank_temperature": rank_temperature,
            # HumanEval has no answer-equality key: two correct programs are almost never identical
            # strings, so PairJudge's team grouping is undefined here and every candidate stands
            # alone. Only read by the pairjudge/rrm rank modes.
            "rm_team_grouping": "none",
        }
    }

    # Pass the official HumanEval prompt directly without SystemMessage or extra instructions
    init: AgentState = {
        "messages": [
            HumanMessage(content=question)
        ],
        "message_intent": "chat",
        "retrieved_context": "",
        "tool_results": "",
        "pool": [],
        "best": None,
        "iteration": 0,
        "max_iterations": 0,
        "width": 1,
        "budget_remaining": 1,
        "final_answer": None,
        "sampled_answers": [],
        "sampled_numbers": [],
        "vote_distribution": {},
        "unique_answers": 0,
    }

    result = await GRAPH_RANKER.ainvoke(init, config=config)
    return result


async def evaluate_sample(
    sample,
    model_name,
    reasoning_mode,
    temperature,
    rank_mode,
    rank_temperature
):
    prompt = sample["prompt"]
    task_id = sample["task_id"]

    try:
        async with SEM:
            state = await solve(
                prompt,
                model_name,
                reasoning_mode,
                temperature,
                rank_mode,
                rank_temperature
            )

        pred_text = state.get("final_answer", "")
        # Clean completion and remove prompt duplication
        extracted_pred = extract_code_completion(pred_text, prompt, sample["entry_point"])

        return {
            "task_id": task_id,
            "completion": extracted_pred,
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "completion": f"# Error during generation: {e}",
        }


async def evaluate(
    dataset, model_name, reasoning_mode, temperature, rank_mode, rank_temperature, jsonl_path
):
    tasks = [
        evaluate_sample(sample, model_name, reasoning_mode, temperature, rank_mode, rank_temperature)
        for sample in dataset
    ]

    results = []
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks))

    for coro in pbar:
        result = await coro
        results.append(result)

    # Save outputs to .jsonl file
    from human_eval.data import write_jsonl
    write_jsonl(jsonl_path, results)
    print(f"\nGenerated completions saved to {jsonl_path}")

    # Safely evaluate correctness using official HumanEval metrics
    from human_eval.evaluation import evaluate_functional_correctness
    print("Running functional correctness evaluation...")
    eval_results = evaluate_functional_correctness(
        sample_file=jsonl_path,
        k=[1],
        n_workers=4,
        timeout=10.0
    )
    print("\n================ RESULTS ================\n")
    for metric_name, val in eval_results.items():
        print(f"{metric_name}: {val:.4f}")
    print("=========================================\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-3b-q4")
    parser.add_argument("--mode", choices=["baseline", "low", "medium", "high", "extra"], default="low")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--rank-mode", choices=["majority", "rank_no_reasoning", "tournament_no_reasoning", "self_certainty_proxy", "compute_matched_usc", "pairjudge", "rrm"], default="tournament_no_reasoning")
    parser.add_argument("--rank-temp", type=float, default=0.0)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--samples", type=int, default=20, help="-1 for full 164 HumanEval problems")
    parser.add_argument("--sem", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    SEM = asyncio.Semaphore(args.sem)

    # Read and convert HumanEval problem map to sequential array slice
    problems = read_problems()
    problem_list = [{"task_id": k, **v} for k, v in problems.items()]

    if args.samples == -1:
        dataset = problem_list
    else:
        dataset = problem_list[:min(args.samples, len(problem_list))]

    if args.out:
        jsonl_path = args.out
        if jsonl_path.endswith(".csv"):
            jsonl_path = jsonl_path[:-4] + ".jsonl"
    else:
        jsonl_path = f"humaneval_{args.model}_{args.mode}_t{args.temp}_n{len(dataset)}.jsonl"

    asyncio.run(
        evaluate(
            dataset=dataset,
            model_name=args.model,
            reasoning_mode=args.mode,
            temperature=args.temp,
            jsonl_path=jsonl_path,
            rank_mode=args.rank_mode,
            rank_temperature=args.rank_temp
        )
    )