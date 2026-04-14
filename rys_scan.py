#!/usr/bin/env python
"""
rys_scan.py — Single-GPU RYS layer-duplication scanner for Qwen3-4B.

Runs all valid (i,j) configs against math_16 + eq_16 probes, saves results
incrementally to JSON. Resume-friendly: skips already-completed configs.

Usage:
    python rys_scan.py                          # full sweep
    python rys_scan.py --only-baseline          # baseline only
    python rys_scan.py --i 15 --j 21            # single config
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# RYS imports (layer duplicator + config only — no queue/fcntl dependency)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "RYS"))
from src.core.layer_duplicator import build_model_with_layers
from src.core.layer_config import baseline_layers, ij_to_layers

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = str(Path(__file__).parent / "Qwen3-4B")
MATH_DATASET = str(Path(__file__).parent / "RYS" / "datasets" / "math_16.json")
EQ_DATASET = str(Path(__file__).parent / "RYS" / "datasets" / "eq_16.json")
RESULTS_DIR = str(Path(__file__).parent / "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "rys_sweep.json")

MATH_MAX_NEW_TOKENS = 64
EQ_MAX_NEW_TOKENS = 128  # EQ answers are short (4 lines); 128 is generous

MATH_SYSTEM_PROMPT = (
    "You are a highly intelligent AI. You have extraordinary intuition and can "
    "easily make accurate estimations. For the following questions, you will "
    "always provide an answer, even if you are not certain."
)

EQ_SYSTEM_PROMPT = ""  # EQ prompts are self-contained

# ---------------------------------------------------------------------------
# Math scoring (from RYS math_worker.py)
# ---------------------------------------------------------------------------

def math_partial_credit(actual: int, estimate: int) -> float:
    """Partial-credit score: digit-level comparison with length penalty."""
    try:
        actual_str = str(int(actual))
        estimate_str = str(int(estimate))
    except (ValueError, OverflowError):
        return 0.0

    max_length = max(len(actual_str), len(estimate_str))
    actual_padded = actual_str.ljust(max_length, "0")
    estimate_padded = estimate_str.ljust(max_length, "0")
    padding_size = max_length - min(len(actual_str), len(estimate_str))

    actual_int = int(actual_padded)
    estimate_int = int(estimate_padded)

    if max(actual_int, estimate_int) == 0:
        return 0.0
    relative_diff = abs(actual_int - estimate_int) / max(actual_int, estimate_int)
    correction_factor = 1 - (padding_size / max_length)
    score = (1 - relative_diff) * correction_factor
    return max(0.0, min(score, 1.0))


def extract_integers(text: str) -> list[int]:
    return [int(p) for p in re.split(r"\D+", text) if p.isdigit()]


def score_math_response(raw_text: str, reference_answer: int) -> float:
    """Score a single math response against reference. Returns best partial credit."""
    # Strip thinking blocks
    stripped = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*$", "", stripped, flags=re.DOTALL).strip()
    integers = extract_integers(stripped)
    if not integers:
        integers = extract_integers(raw_text)
    if not integers:
        return 0.0
    return max(math_partial_credit(reference_answer, i) for i in integers)


# ---------------------------------------------------------------------------
# EQ scoring (from RYS eq_worker.py)
# ---------------------------------------------------------------------------

EMOTION_KEYS = ["emotion1_score", "emotion2_score", "emotion3_score", "emotion4_score"]


def extract_scores_from_section(text: str) -> list[float] | None:
    """Extract up to 4 numeric scores (0-10) from text."""
    score_pattern = r"(?:\d\.\s*)?[A-Za-z]+:\s*(\d+(?:\.\d+)?)"
    matches = re.findall(score_pattern, text)
    valid = []
    for m in matches[:4]:
        try:
            val = float(m)
            if 0 <= val <= 10:
                valid.append(val)
        except ValueError:
            continue
    if len(valid) >= 3:
        while len(valid) < 4:
            valid.append(5.0)
        return valid
    return None


def extract_emotion_scores(text: str) -> tuple[dict | None, float]:
    """Extract emotion scores dict from model output. Returns (scores, confidence)."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*$", "", stripped, flags=re.DOTALL).strip()

    # Try "First pass scores:" section
    match = re.search(r"First pass scores:", stripped, re.IGNORECASE)
    if match:
        after = stripped[match.end():]
        end = re.search(r"\[End of answer\]", after, re.IGNORECASE)
        if end:
            after = after[: end.start()]
        scores = extract_scores_from_section(after)
        if scores:
            return (
                {k: v for k, v in zip(EMOTION_KEYS, scores)},
                1.0,
            )

    # Fallback: any 4 numbers in 0-10 range
    all_nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", stripped)
    valid = [float(n) for n in all_nums if 0 <= float(n) <= 10][:4]
    if len(valid) >= 3:
        while len(valid) < 4:
            valid.append(5.0)
        return {k: v for k, v in zip(EMOTION_KEYS, valid)}, 0.5

    return None, 0.0


def score_eq_response(raw_text: str, reference: dict) -> float:
    """Score a single EQ response. Returns 0-1 score."""
    predicted, confidence = extract_emotion_scores(raw_text)
    if predicted is None:
        return 0.5  # neutral baseline for failed extraction

    pred_scores = [predicted.get(k, 5.0) for k in EMOTION_KEYS]
    ref_scores = [reference.get(k, 5.0) for k in EMOTION_KEYS]

    total_diff = sum(abs(p - r) for p, r in zip(pred_scores, ref_scores))
    max_diff = 10 * 4
    raw_score = max(0.0, 1.0 - total_diff / max_diff)
    return confidence * raw_score + (1 - confidence) * 0.5


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(model, input_ids, attention_mask, max_new_tokens, pad_token_id):
    """Generate greedy completion, return only new token ids."""
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    return out[0][input_ids.shape[1]:]


def is_garbage(text: str) -> bool:
    """Detect clearly broken output (repetitive tokens, no alphanumeric content)."""
    stripped = text.strip()
    if len(stripped) < 2:
        return True
    # If less than 10% alphanumeric, it's garbage
    alnum = sum(c.isalnum() for c in stripped)
    if alnum / max(len(stripped), 1) < 0.10:
        return True
    return False


# ---------------------------------------------------------------------------
# Dataset loading + pre-tokenization
# ---------------------------------------------------------------------------

def load_and_tokenize_math(path, tokenizer, device):
    with open(path) as f:
        raw = json.load(f)
    items = []
    for qid, sample in raw.items():
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": f"/no_think {sample['question']}"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = tokenizer(text, return_tensors="pt")
        items.append({
            "qid": qid,
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
            "answer": sample["answer"],
        })
    return items


def load_and_tokenize_eq(path, tokenizer, device):
    with open(path) as f:
        raw = json.load(f)
    items = []
    for qid, sample in raw.items():
        messages = [
            {"role": "user", "content": f"/no_think {sample['prompt']}"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = tokenizer(text, return_tensors="pt")
        items.append({
            "qid": qid,
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
            "reference": sample["reference_answer"],
        })
    return items


# ---------------------------------------------------------------------------
# Evaluate one config
# ---------------------------------------------------------------------------

def evaluate_config(
    model, layer_indices, math_items, eq_items, tokenizer, num_layers
):
    """Run math + EQ probes on a layer config. Returns dict with scores."""
    is_baseline = list(layer_indices) == list(range(num_layers))

    if is_baseline:
        wrapped = model
    else:
        wrapped = build_model_with_layers(model, layer_indices)

    pad_id = tokenizer.eos_token_id

    # --- Math ---
    math_scores = []
    math_details = []
    zero_count = 0
    for item in math_items:
        gen_ids = generate_one(
            wrapped, item["input_ids"], item["attention_mask"],
            MATH_MAX_NEW_TOKENS, pad_id
        )
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        sc = score_math_response(raw, item["answer"])
        math_scores.append(sc)
        math_details.append({"qid": item["qid"], "output": raw, "score": sc})
        # Track zero-scoring probes
        if sc == 0.0 or is_garbage(raw):
            zero_count += 1
        # Early stop: if first 2 probes both score zero/garbage, config is broken
        if zero_count >= 2 and len(math_scores) <= 2:
            math_scores.extend([0.0] * (len(math_items) - len(math_scores)))
            eq_scores = [0.5] * len(eq_items)
            eq_details = [{"qid": it["qid"], "output": "[skipped]", "score": 0.5} for it in eq_items]
            math_avg = sum(math_scores) / len(math_scores)
            return {
                "math_score": round(math_avg, 6),
                "eq_score": 0.5,
                "combined": round(math_avg + 0.5, 6),
                "num_layers_total": len(layer_indices),
                "extra_layers": len(layer_indices) - num_layers,
                "overhead_pct": round(100 * (len(layer_indices) - num_layers) / num_layers, 2),
                "math_details": math_details,
                "eq_details": eq_details,
                "early_stopped": True,
            }
    # --- EQ ---
    eq_scores = []
    eq_details = []
    for item in eq_items:
        gen_ids = generate_one(
            wrapped, item["input_ids"], item["attention_mask"],
            EQ_MAX_NEW_TOKENS, pad_id
        )
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        sc = score_eq_response(raw, item["reference"])
        eq_scores.append(sc)
        eq_details.append({"qid": item["qid"], "output": raw, "score": sc})

    math_avg = sum(math_scores) / len(math_scores) if math_scores else 0.0
    eq_avg = sum(eq_scores) / len(eq_scores) if eq_scores else 0.0

    return {
        "math_score": round(math_avg, 6),
        "eq_score": round(eq_avg, 6),
        "combined": round(math_avg + eq_avg, 6),
        "num_layers_total": len(layer_indices),
        "extra_layers": len(layer_indices) - num_layers,
        "overhead_pct": round(100 * (len(layer_indices) - num_layers) / num_layers, 2),
        "math_details": math_details,
        "eq_details": eq_details,
    }


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)


def config_key(i, j):
    return f"{i},{j}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_configs(num_layers):
    """Generate all valid (i,j) pairs including baseline (0,0)."""
    configs = [(0, 0)]  # baseline first
    for j in range(1, num_layers + 1):
        for i in range(j):
            configs.append((i, j))
    return configs


def main():
    parser = argparse.ArgumentParser(description="RYS single-GPU scanner")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--math-dataset", default=MATH_DATASET)
    parser.add_argument("--eq-dataset", default=EQ_DATASET)
    parser.add_argument("--results-file", default=RESULTS_FILE)
    parser.add_argument("--only-baseline", action="store_true")
    parser.add_argument("--i", type=int, default=None, help="Single config start")
    parser.add_argument("--j", type=int, default=None, help="Single config end")
    args = parser.parse_args()

    device = "cuda:0"

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=device, local_files_only=True
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers, {torch.cuda.max_memory_allocated()/1e9:.1f} GB VRAM")

    # Pre-tokenize probes
    print("Pre-tokenizing math probes...")
    math_items = load_and_tokenize_math(args.math_dataset, tokenizer, device)
    print(f"  {len(math_items)} math questions")

    print("Pre-tokenizing EQ probes...")
    eq_items = load_and_tokenize_eq(args.eq_dataset, tokenizer, device)
    print(f"  {len(eq_items)} EQ scenarios")

    # Determine configs to run
    if args.i is not None and args.j is not None:
        configs = [(args.i, args.j)]
    elif args.only_baseline:
        configs = [(0, 0)]
    else:
        configs = generate_all_configs(num_layers)

    print(f"Total configs to evaluate: {len(configs)}")

    # Load existing results for resume
    results = load_results(args.results_file)
    skip_count = 0

    t_start = time.time()
    for idx, (i, j) in enumerate(configs):
        key = config_key(i, j)
        if key in results:
            skip_count += 1
            continue

        # Compute layer indices
        if i == 0 and j == 0:
            layer_indices = baseline_layers(num_layers)
            label = "baseline"
        else:
            layer_indices = ij_to_layers(num_layers, i, j)
            label = f"({i},{j}) +{len(layer_indices)-num_layers} layers"

        # Progress
        done = idx + 1 - skip_count
        remaining = len(configs) - idx - 1
        elapsed = time.time() - t_start
        rate = elapsed / done if done > 0 else 0
        eta = rate * remaining if done > 0 else 0
        print(
            f"[{idx+1}/{len(configs)}] {label:25s}  "
            f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
            end="",
            flush=True,
        )

        t0 = time.time()
        result = evaluate_config(
            model, layer_indices, math_items, eq_items, tokenizer, num_layers
        )
        dt = time.time() - t0

        result["wall_seconds"] = round(dt, 2)
        result["layer_indices"] = layer_indices
        results[key] = result

        # Save after every config (resume-friendly)
        save_results(args.results_file, results)

        print(
            f"  math={result['math_score']:.4f}  eq={result['eq_score']:.4f}  "
            f"combined={result['combined']:.4f}  ({dt:.1f}s)"
        )

    total_time = time.time() - t_start
    print(f"\nDone. {len(results)} configs total. Skipped {skip_count} (already done).")
    print(f"Total wall time: {total_time/60:.1f} min")
    print(f"Results saved to: {args.results_file}")


if __name__ == "__main__":
    main()
