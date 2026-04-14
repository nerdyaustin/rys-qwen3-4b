# RYS Layer Duplication Experiment — Qwen3-4B

## Objective

Test whether inference-time repetition of contiguous middle transformer layers
improves math and emotional-quotient (EQ) probe scores on a small dense model,
replicating David Noel Ng's RYS methodology at 4B scale on consumer hardware.

Core question: **Does the three-phase anatomy (encode / reason / decode)
exist at 4B scale, and can the reasoning region be profitably re-traversed?**

---

## References

| Source | Link |
|--------|------|
| RYS repo | github.com/dnhkng/RYS |
| LLM Neuroanatomy I | dnhkng.github.io/posts/rys/ |
| LLM Neuroanatomy II | dnhkng.github.io/posts/rys-ii/ |
| Reasoning with Latent Thoughts | arxiv.org/abs/2502.17416 |
| LoopFormer | arxiv.org/abs/2602.11451 |
| Handoff doc | D:/LLM/looped_llm_nanogpt_handoff_v2.md |

---

## Hardware

- **GPU**: NVIDIA RTX 3090, 24 GB VRAM
- **OS**: Windows 10
- **Python**: 3.12, PyTorch 2.5.1+cu121, transformers 5.3.0.dev0

---

## Model: Qwen3-4B

- **Architecture**: Qwen3ForCausalLM (pure dense, standard self-attention)
- **Layers**: 36
- **Hidden size**: 2560
- **Attention**: GQA — 32 Q heads, 8 KV heads, head_dim 128
- **Positional encoding**: RoPE (theta=1e6)
- **Params**: 4.0B (3.6B non-embedding)
- **Dtype**: BF16
- **VRAM footprint**: ~8.1 GB peak (baseline generation)

### Why this model

- Pure dense transformer — no hybrid linear/full attention (unlike Qwen3.5-2B)
- 36 layers — enough depth for interesting anatomy, 631 (i,j) sweep configs
- Fits comfortably on 3090 with room for KV cache overhead from duplication
- Same model family as Ng's Qwen3.5-27B target, enabling cross-scale comparison

---

## Probes

### Math (math_16.json)

16 questions: square roots, cube roots, multiplication, multi-op combinations.
All require single-integer output — no chain-of-thought, no tool use.
Scoring: partial credit based on digit-level accuracy (Ng's custom scorer).

### EQ (eq_16.json)

16 EQ-Bench scenarios: complex social dialogues, predict 4 emotion intensities (0-10).
Scoring: Pearson correlation between predicted and reference emotion scores.

### Decoding config

- `do_sample=False` (greedy) — verified perfectly deterministic across 5 runs
- `max_new_tokens=64` (math), `max_new_tokens=256` (EQ)
- System prompt: Ng's standard math system prompt
- `/no_think` prefix to disable Qwen3 thinking mode

---

## Experimental Design

### RYS (i,j) convention

For a model with N layers, config `(i, j)` means:
- Execute layers `0..j-1` normally
- Jump back and re-execute layers `i..N-1`
- Layers `i..j-1` are traversed twice
- `(0, 0)` = baseline (unmodified)

For N=36: 631 valid (i,j) pairs including baseline.

### Sweep plan

1. Baseline (0,0)
2. All valid (i,j) for i < j, j <= 36
3. Record math score + EQ score per config
4. Generate heatmaps showing delta from baseline

---

## Results

### Preliminary observations (pre-sweep)

Quick manual test on "What is 7+13?":
- **Baseline (0,0)**: "7 + 13 = 20" (correct)
- **Middle dup (15,21)**: "7 + 13 equals 20." (correct, different phrasing)
- **Early dup (0,6)**: garbage (blank lines)
- **Late dup (30,36)**: garbage (`.urlopen`, `literal`, etc.)

Three-phase anatomy clearly visible even on a single trivial question.

### Determinism check

5 runs of math probe 0 with greedy decoding: **all identical** (output: "977563233").
Answer wrong (expected 596252164) but partial credit scorer handles this.
No need for multi-run variance testing with greedy.

### Baseline scores

| Probe | Score | Notes |
|-------|-------|-------|
| Math-16 | 0.3047 | Partial credit — model gets some digits right on very hard math |
| EQ-16 | 0.7490 | Strong signal — model is decent at predicting emotion intensities |
| Combined | 1.0537 | |

### Preliminary spot-checks

| Config | Math | EQ | Combined | Delta vs baseline | Time |
|--------|------|-----|----------|------------------|------|
| Baseline (0,0) | 0.3047 | 0.7490 | 1.0537 | — | 30s |
| Middle (15,21) | **0.3776** | **0.7549** | **1.1325** | **+0.0788** | 36s |
| Early (0,6) | 0.0000 | 0.5000 | 0.5000 | -0.5537 | 160s (garbage) |
| Late (30,36) | — | — | — | — | timed out (garbage) |

Middle block duplication shows ~24% improvement on math and slight EQ gain.
Early/late blocks produce catastrophic failure — exactly matching three-phase anatomy prediction.

### Sweep results

### Sweep results (667/667 configs — COMPLETE)

| Metric | Best config | Score | Delta | Extra layers | Overhead % |
|--------|------------|-------|-------|-------------|-----------|
| Math | (21,23) | 0.4471 | +0.1424 | 2 | 5.6 |
| EQ | (7,14) | 0.8157 | +0.0667 | 7 | 19.4 |
| Combined | (14,36) | 1.2016 | +0.1479 | 22 | 61.1 |

151/666 non-baseline configs beat baseline (22.7%).
115 configs early-stopped (garbage detection).

Three distinct hot zones in heatmap:
1. **Zone A**: i~7-11, j~13-16 — layers 7-15 region (~19-44% depth)
2. **Zone B**: i~15-19, j~18-21 — layers 15-20 region (~42-58% depth)
3. **Zone C**: i~21-24, j~23-27 — layers 21-26 region (~58-75% depth)

Full-width repeats (j=36) from mid-stack onward also productive:
(14,36)=1.2016, (10,36)=1.1684, (7,36)=1.1648 — but at 60-80% overhead.

### Single-layer repeat champions

14 of 35 single-layer repeats beat baseline. Strong pattern:

| Layer | Config | Combined delta | Notes |
|-------|--------|---------------|-------|
| 21 | (21,22) | +0.1257 | Best single-layer overall |
| 5 | (5,6) | +0.1007 | Early reasoning boundary |
| 24 | (24,25) | +0.1003 | Late reasoning region |
| 26 | (26,27) | +0.0726 | |
| 19 | (19,20) | +0.0708 | |
| 22 | (22,23) | +0.0627 | |
| 20 | (20,21) | +0.0565 | |
| 17 | (17,18) | +0.0493 | |

Contrast with Ng's 27B result: single-layer repeats "almost never help."
At 4B, they work consistently across a broad range (layers 5-26).
This may reflect the smaller model's more diffuse functional anatomy.

### Pareto frontier (final)

| Size | Config (i,j) | Extra layers | Overhead % | Math | EQ | Combined | Delta |
|------|-------------|-------------|-----------|------|-----|----------|-------|
| XS | (2,3) | 1 | 2.8 | 0.3485 | 0.7412 | 1.0898 | +3.4% |
| S | (5,6) | 1 | 2.8 | 0.4104 | 0.7441 | 1.1545 | +9.6% |
| M | (21,22) | 1 | 2.8 | 0.4313 | 0.7481 | 1.1794 | +11.9% |
| L | (21,23) | 2 | 5.6 | 0.4471 | 0.7453 | 1.1924 | +13.2% |
| XL | (14,36) | 22 | 61.1 | 0.4134 | 0.7882 | 1.2016 | +14.0% |

**(21,22) is the practical efficiency champion**: +11.9% combined improvement
from repeating just layer 21 once, at only 2.8% overhead. The jump from
(21,23) at +13.2% to (14,36) at +14.0% costs 10x the overhead — steep
diminishing returns above 2 extra layers.

---

## Comparison with Ng's Qwen3.5-27B findings

Ng's Pareto-optimal configs for 27B (64 layers):
- S: layer 33 (+1.5%)
- M: layers 31-33 (+4.7%)
- L: layers 29-35 (+10.9%)
- XL: layers 26-35 (+15.6%)

Sweet spot was consistently mid-stack, around layers 26-35 out of 64 (~40-55% depth).

**Our Qwen3-4B findings (36 layers):**
- Three hot zones: layers 7-15, 15-20, 21-26
- Best efficiency configs cluster around layer 21 (~58% depth)
- Best absolute config (14,36) repeats layers 14-35 at 61% overhead
- Single-layer repeats work consistently at 4B scale (14/35 beat baseline)
  vs rarely at 72B — smaller model has more diffuse functional anatomy
- The "reasoning region" is broader relative to total depth
- The blue wall (encoding zone) spans ~layers 0-6 (~0-17% depth)
- Strong individual repeat layers: 5, 17-26 — nearly the entire mid-to-late stack
- Ng's 27B saw +15.6% max at 15.6% overhead; our 4B sees +11.9% at 2.8% overhead
  and +14.0% max at 61% overhead — steeper diminishing returns at small scale

### Key difference: scale affects RYS efficiency curve shape

At 27B, adding more repeated layers yields roughly linear improvement up to ~10
extra layers. At 4B, the curve is sharply concave — most of the benefit comes
from the first 1-2 extra layers, with massive overhead required to push further.
This suggests larger models have more specialized layers where repetition compounds,
while smaller models benefit from a single extra pass through their most general-purpose
reasoning layer.

---

## Files

| File | Purpose |
|------|---------|
| `D:/LLM/rys_experiment_notes.md` | This file — experiment log |
| `D:/LLM/rys_scan.py` | Scanner script |
| `D:/LLM/rys_analyze.py` | Heatmap + analysis script |
| `D:/LLM/results/` | Output directory for sweep data |
| `D:/LLM/RYS/` | Cloned RYS repo (layer duplicator, probes) |
| `D:/LLM/Qwen3-4B/` | Model weights |
