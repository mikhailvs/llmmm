# Epicure Implementation Plan

Synthesized from four reviewer streams (Culinary Scientist, ML Engineer, CLI/UX, Home Cook).
Items flagged with **[MULTI]** were independently requested by 2+ reviewers and receive priority boost.

---

## Phase 1 — High impact, easy (this weekend, ~1 developer day)

### 1. Underscore → space in display output **[MULTI: UX + Home Cook]**

**What it does:** Shows "bay leaf" instead of "bay_leaf" in all human-facing output. Pipe output stays machine-readable (underscored).

**How:** Add a one-line helper and apply it everywhere `PIPE_OUT` is False.

```python
def display(name: str) -> str:
    return name.replace("_", " ")
```

Apply in `cmd_pair`, `cmd_similar`, `cmd_steer`, `cmd_search`: wrap every printed ingredient name with `display()`. Pipe paths (`print(name)`) stay unchanged.

**Estimated lines:** ~3 lines new code + 8 call-site changes.

---

### 2. `--count N` flag **[UX]**

**What it does:** Controls how many results are returned. Default stays 12 for pair/similar, 6 per tier for steer.

**How:** Add a global pre-parser step before dispatching to commands. Parse `--count N` from `sys.argv` before passing remaining args to the command handler; thread `k` into `top_k()` calls.

```python
# In main, before dispatch:
count = 12
if "--count" in sys.argv:
    i = sys.argv.index("--count")
    count = int(sys.argv[i + 1])
    sys.argv = sys.argv[:i] + sys.argv[i+2:]
```

Pass `count` as an argument to `cmd_pair`, `cmd_similar`, etc., replacing the hardcoded `k=12`/`k=15`.

**Estimated lines:** ~10 lines.

---

### 3. `--embedding {cooc,core,chem}` flag + default `pair` to `chem` **[MULTI: Culinary + ML]**

**What it does:** Exposes all three embeddings. `pair` and `similar` default to `chem` (flavor compound overlap); `recipe` and `steer` default to `cooc` (cultural co-occurrence). User can override with `--embedding cooc`.

**How:** Pre-parse `--embedding` from `sys.argv` the same way as `--count`. Map to file paths and pass the chosen `names, vecs` to each command. The per-command defaults require a small dict lookup.

```python
EMBEDDING_FILES = {
    "cooc": "epicure_cooc.csv",
    "chem": "epicure_chem.csv",
    "core": "epicure_core.csv",
}
COMMAND_DEFAULT_EMBEDDING = {
    "pair": "chem", "similar": "chem",
    "steer": "cooc", "recipe": "cooc",
    "search": "cooc", "sub": "cooc",
    "profile": "chem",
}
```

All three CSVs share the same 1,790 ingredient names in the same row order, so the rest of the code is unaffected.

**Estimated lines:** ~20 lines.

---

### 4. `sub <ingredient>` substitution command **[MULTI: Culinary + Home Cook]**

**What it does:** Finds the nearest neighbors to an ingredient in cooc space — functional substitutes with similar culinary role. Optional `--avoid <tag>` placeholder reserved for Phase 2.

**How:** Thin wrapper around existing `top_k`. Returns top-5 most similar by cosine similarity, excluding the query itself.

```python
def cmd_sub(args, names, vecs):
    args = merge_args(args)
    if not args:
        print("Usage: epicure.py sub <ingredient>", file=sys.stderr); return
    n = resolve_one(args[0], names)
    if not n: return
    idx = names.index(n)
    results = top_k(vecs[idx], vecs, names, k=6, exclude=[idx])
    if PIPE_OUT:
        for name, _ in results: print(name)
    else:
        print(f"\nSubstitutes for '{display(n)}':")
        print("─" * 50)
        for name, score in results:
            print(f"  {display(name):<30} {score:.3f}")
```

Add `"sub": cmd_sub` to `COMMANDS` and one line to `HELP`.

**Estimated lines:** ~20 lines.

---

### 5. `fridge` command alias (no-pipe recipe) **[MULTI: Home Cook + UX]**

**What it does:** `epicure.py fridge chicken onion lemon` immediately generates a recipe — the most common home cook flow without requiring pipe knowledge.

**How:** `fridge` is an alias for `recipe` that also sets `PIPE_OUT = False` behaviour so the user sees the full formatted output. Implementation is a one-line alias in the dispatch table.

```python
COMMANDS["fridge"] = cmd_recipe   # identical behavior; recipe already handles direct args
```

Update `HELP` to document `fridge` as the friendly no-pipe shortcut.

**Estimated lines:** ~5 lines (alias + help text).

---

### 6. `list` discovery commands **[UX]**

**What it does:** `epicure.py list cuisines` and `epicure.py list ingredients [filter]` let users discover what's available without needing to know the vocabulary.

**How:**

```python
def cmd_list(args, names, vecs):
    if not args or args[0] == "cuisines":
        print("Available cuisines:")
        for c in CUISINE_MARKERS: print(f"  {c}")
    elif args[0] == "ingredients":
        term = normalize(args[1]) if len(args) > 1 else ""
        matches = [n for n in names if term in n] if term else names
        for m in matches:
            print(display(m) if not PIPE_OUT else m)
```

**Estimated lines:** ~15 lines.

---

### 7. `--serves N` and `--time N` flags injected into recipe prompt **[MULTI: Culinary + Home Cook]**

**What it does:** Passes serving size and time constraint to the LLM. The culinary note about non-linear salt/acid/spice scaling is included automatically when `--serves` differs from default.

**How:** Pre-parse `--serves` and `--time` from argv. Inject into the prompt string in `cmd_recipe`:

```python
constraints = []
if serves: constraints.append(
    f"Scale for {serves} servings. Note: salt, acid, and spice do NOT scale linearly — "
    f"use roughly 70% of the per-serving amount when scaling up.")
if time: constraints.append(f"The total cook time must be under {time} minutes.")
# Append constraints to prompt before final line
```

**Estimated lines:** ~15 lines.

---

### 8. Dietary flags (`--vegan`, `--gluten-free`, `--dairy-free`) injected into recipe prompt **[Home Cook]**

**What it does:** Adds a constraint line to the recipe LLM prompt. Pure prompt engineering — no embedding logic needed.

**How:**

```python
DIETARY_FLAGS = {"--vegan": "vegan (no meat, fish, dairy, eggs, or honey)",
                 "--gluten-free": "gluten-free", "--dairy-free": "dairy-free"}
# Pre-parse flags; build dietary_str; inject as:
# f"The recipe must be {dietary_str}."
```

**Estimated lines:** ~12 lines.

---

## Phase 2 — High impact, moderate work (~2–4 days)

### 9. `profile` subcommand: flavor balance breakdown **[Culinary Scientist]**

**What it does:** Projects the centroid of a set of ingredients onto six flavor poles (umami, acid, fat, sweet, bitter, heat) and prints a scored bar chart. Also usable as a pre-check before `recipe`.

**How:** Precompute six pole vectors as normalized centroids of anchor sets (all anchors confirmed present in vocabulary):

```python
FLAVOR_ANCHORS = {
    "umami":  ["miso", "soy_sauce", "parmesan_cheese", "anchovy", "mushroom", "tomato"],
    "acid":   ["lemon", "vinegar", "lime", "tamarind", "yogurt", "tomato"],
    "fat":    ["butter", "olive_oil", "cream", "lard", "coconut_milk", "avocado"],
    "sweet":  ["honey", "sugar", "maple_syrup", "vanilla", "mirin", "date"],
    "bitter": ["coffee", "radicchio", "arugula", "turmeric", "tonic_water"],
    "heat":   ["chili_pepper", "black_pepper", "ginger", "horseradish", "wasabi", "jalapeno"],
}
```

Compute poles once at load time (or lazily on first `profile` call). Score = dot product of ingredient centroid with each pole. Scores range ~0.35–0.65 in practice; normalize to 0–1 for display using `(score - 0.35) / 0.30`.

Pipe output emits `flavor:score` pairs for scripting.

**--json output:** `{"umami": 0.72, "acid": 0.55, ...}`

**Estimated lines:** ~45 lines.

---

### 10. Flavor balance injection into `recipe` prompt **[Culinary Scientist]**

**What it does:** Runs the `profile` logic internally before calling Ollama, identifies the weakest flavor dimension(s), and tells the LLM to balance them using the suggested complements.

**How:** Reuse the pole vectors from item 9. After computing complements, check which flavor poles score below a threshold (e.g., normalized score < 0.3). Build a hint string:

```python
weak = [f for f, score in profile_scores.items() if score < threshold]
if weak:
    hint = f"The current ingredients are low in {', '.join(weak)}. "
    hint += "Use the suggested pairings and technique to bring balance."
    prompt += f"\n\nFlavor note: {hint}"
```

**Depends on:** Item 9 (flavor pole vectors).

**Estimated lines:** ~20 lines (reuses pole code from item 9).

---

### 11. Hubness penalty in `top_k` **[ML Engineer]**

**What it does:** Prevents ubiquitous ingredients (salt, black_pepper, olive_oil) from dominating every result by penalizing ingredients with high average similarity across the corpus.

**How:** Precompute `hub_score[i] = mean cosine similarity of ingredient i to all others` at load time. This is `(vecs @ vecs.T).mean(axis=1)` — O(n²) but n=1,790 finishes in ~50ms. Add an optional penalty parameter to `top_k`:

```python
def top_k(query_vec, vecs, names, k=12, exclude=None, hub_scores=None, hub_alpha=0.5):
    sims = vecs @ query_vec
    if hub_scores is not None:
        sims = sims - hub_alpha * hub_scores
    ...
```

Enable by default for `pair`, `similar`, `profile`; disabled for `steer` (where hub ingredients may be legitimately useful as cuisine anchors).

**Estimated lines:** ~15 lines + 1 load-time computation.

---

### 12. Tension detection warning **[Culinary Scientist + ML Engineer]**

**What it does:** When the minimum pairwise cosine similarity between input ingredients falls below a threshold (~0.05, validated experimentally), prints a warning and passes the tension to the recipe prompt.

**How:** In `cmd_pair` and `cmd_recipe`, after resolving indices:

```python
def detect_tensions(indices, vecs, names, threshold=0.05):
    tensions = []
    for i, a in enumerate(indices):
        for b in indices[i+1:]:
            sim = float(vecs[a] @ vecs[b])
            if sim < threshold:
                tensions.append((names[a], names[b], sim))
    return tensions
```

In terminal output, print a `[!] Flavor tension: chocolate and beef are rarely used together` warning. In the recipe prompt, add: `"These ingredients are unconventional together: {pairs}. Acknowledge this and create a creative dish that bridges the gap."`

**Estimated lines:** ~25 lines.

---

### 13. `--json` output flag **[UX]**

**What it does:** All commands emit structured JSON when `--json` is passed, enabling downstream scripting without parsing human-readable text.

**How:** Pre-parse `--json` flag. Each command builds a result dict instead of printing directly, then `json.dumps(result)` at the end. Commands that currently print progressively (pair, similar, sub) return `{"results": [{"name": ..., "score": ...}]}`. `profile` returns `{"profile": {"umami": 0.72, ...}}`. `recipe` streams to stderr and emits `{"recipe": "...full text..."}` to stdout.

Requires refactoring each command to return a data structure rather than print inline — the largest structural change in Phase 2.

**Estimated lines:** ~40 lines (mostly mechanical refactor).

---

### 14. Better error messages with "did you mean?" suggestions **[UX]**

**What it does:** When an ingredient is not found, the current code silently skips it. This replaces that with a targeted suggestion using the existing `resolve()` function.

**How:** In `resolve_one`, when the exact match fails, `resolve()` already computes fuzzy candidates. Surface the top candidate more prominently:

```python
if not candidates:
    print(f"  [!] '{name}' not found.", file=sys.stderr)
    # Find closest by edit distance as fallback
    from difflib import get_close_matches
    close = get_close_matches(normalize(name), names, n=3, cutoff=0.6)
    if close:
        print(f"      Did you mean: {', '.join(display(c) for c in close)}?", file=sys.stderr)
    return None
```

**Estimated lines:** ~8 lines.

---

### 15. REPL / interactive mode **[UX]**

**What it does:** `python epicure.py` with no arguments drops into a prompt loop, letting users explore without re-loading embeddings on each command.

**How:** In `__main__`, when `len(sys.argv) < 2` and `not PIPE_IN`, enter a loop instead of printing help:

```python
def repl(names, vecs):
    import readline  # enables history on Linux/macOS
    print("epicure REPL — type 'help' or a command. Ctrl-D to exit.\n")
    while True:
        try:
            line = input("epicure> ").strip()
        except EOFError:
            break
        if not line: continue
        parts = line.split()
        cmd, rest = parts[0].lower(), parts[1:]
        if cmd in COMMANDS:
            COMMANDS[cmd](rest, names, vecs)
        elif cmd in ("quit", "exit"):
            break
        else:
            print(f"Unknown command '{cmd}'. Commands: {', '.join(COMMANDS)}")
```

Embeddings load once at REPL entry. No `PIPE_OUT` complications because the REPL is always interactive.

**Estimated lines:** ~25 lines.

---

## Phase 3 — Nice to have (polish, longer research)

### 16. IDF-weighted cuisine poles **[ML Engineer]**

**What it does:** Replaces equal-weight marker averaging with IDF weighting, so rare cuisine-specific ingredients (e.g., `za_atar` for Middle Eastern) count more than ubiquitous ones (garlic appears in 8 of 10 cuisines).

**How:** Count how many cuisine lists each marker appears in. Weight = `log(N_cuisines / count + 1)`.

```python
from collections import Counter
marker_freq = Counter(m for markers in CUISINE_MARKERS.values() for m in markers)
n_cuisines = len(CUISINE_MARKERS)
def idf_weight(marker):
    return np.log(n_cuisines / marker_freq[marker] + 1)
```

Apply weights in `cuisine_vector()` instead of `vecs[indices].mean()`.

**Estimated lines:** ~15 lines.

---

### 17. Weighted centroid (downweight outlier ingredients) **[ML Engineer]**

**What it does:** When aggregating a set of ingredients, downweights ingredients that are semantically distant from the group centroid — reduces noise when users pass a diverse or contradictory set.

**How:** Iterative re-weighting: compute uniform centroid, compute per-ingredient similarity to centroid, re-weight by similarity, recompute. One or two iterations suffice.

```python
def weighted_centroid(indices, vecs, iterations=2):
    v = vecs[indices].mean(axis=0)
    v /= np.linalg.norm(v)
    for _ in range(iterations):
        weights = np.clip(vecs[indices] @ v, 0.1, 1.0)
        v = (vecs[indices] * weights[:, None]).sum(axis=0)
        v /= np.linalg.norm(v)
    return v
```

Replace `centroid()` calls in `cmd_pair`, `cmd_recipe`, `cmd_steer`.

**Estimated lines:** ~15 lines.

---

### 18. Embedding arithmetic: "X but more Y" steer mode **[ML Engineer]**

**What it does:** `epicure.py steer --push umami salmon` computes `salmon_vec + (umami_pole - mean)` to find ingredients that are "salmon-like but with more umami character."

**How:** Reuse flavor poles from item 9. Add `--push <flavor>` option to `steer` command:

```python
# steered = ingredient_vec + alpha * (flavor_pole - mean_vec)
mean_vec = vecs.mean(axis=0)
mean_vec /= np.linalg.norm(mean_vec)
steered = base + alpha * (poles[push_flavor] - mean_vec)
steered /= np.linalg.norm(steered)
```

**Depends on:** Item 9 (flavor poles).

**Estimated lines:** ~20 lines.

---

### 19. Technique tagging in recipe prompt **[Culinary Scientist]**

**What it does:** Injects a small hint about best cooking technique per ingredient (e.g., "braise: beef short rib, roast: beets, raw: cucumber") into the recipe prompt.

**How:** Hardcoded lookup of ~100 ingredients to `{raw, roast, braise, fry, steam, grill, simmer}`. Falls back to no annotation. Inject as a parenthetical in the prompt's ingredient list.

**Estimated lines:** ~30 lines (mostly the lookup table).

---

### 20. Data-driven cuisine centroids from embedding density **[Culinary Scientist + ML Engineer]**

**What it does:** Replaces the hardcoded `CUISINE_MARKERS` lists with automatically computed centroids derived from clusters of ingredients that are densely connected in cooc space. Requires the user to have access to a cuisine-labeled ingredient dataset or to run a one-time offline clustering step.

**How:** This requires external labeled data or a semi-supervised approach (e.g., k-means initialized from current markers, then expanded). Out of scope for a weekend — flag as a future research item.

**Estimated lines:** New offline script (~80 lines) + 5-line change to load precomputed centroids.

---

## Summary Table

| # | Feature | Phase | Reviewers | Est. Lines |
|---|---------|-------|-----------|------------|
| 1 | Underscore → space display | 1 | UX, Home Cook | ~11 |
| 2 | `--count N` flag | 1 | UX | ~10 |
| 3 | `--embedding` flag + chem default for pair | 1 | Culinary, ML | ~20 |
| 4 | `sub` substitution command | 1 | Culinary, Home Cook | ~20 |
| 5 | `fridge` alias command | 1 | Home Cook, UX | ~5 |
| 6 | `list cuisines/ingredients` | 1 | UX | ~15 |
| 7 | `--serves N` / `--time N` prompt injection | 1 | Culinary, Home Cook | ~15 |
| 8 | Dietary flags | 1 | Home Cook | ~12 |
| 9 | `profile` flavor breakdown command | 2 | Culinary | ~45 |
| 10 | Flavor balance in recipe prompt | 2 | Culinary | ~20 |
| 11 | Hubness penalty | 2 | ML | ~15 |
| 12 | Tension detection | 2 | Culinary, ML | ~25 |
| 13 | `--json` output flag | 2 | UX | ~40 |
| 14 | "Did you mean?" error messages | 2 | UX | ~8 |
| 15 | REPL / interactive mode | 2 | UX | ~25 |
| 16 | IDF-weighted cuisine poles | 3 | ML | ~15 |
| 17 | Weighted centroid (outlier downweight) | 3 | ML | ~15 |
| 18 | Embedding arithmetic `--push <flavor>` | 3 | ML | ~20 |
| 19 | Technique tagging in recipe prompt | 3 | Culinary | ~30 |
| 20 | Data-driven cuisine centroids | 3 | Culinary, ML | ~85 |

**Phase 1 total:** ~108 lines added to a 371-line file. All items are independent and can be done in any order.

**Phase 2 total:** ~178 lines. Items 10 and 18 depend on item 9's pole vectors; everything else is independent.

---

## Implementation Order for a Weekend Sprint

**Saturday morning (~3h):** Items 1, 2, 3, 5, 6 — all mechanical, no new math.

**Saturday afternoon (~3h):** Items 4, 7, 8 — new commands and prompt flags, still no new math.

**Sunday morning (~3h):** Items 9 + 10 together (poles + recipe injection), item 11 (hubness).

**Sunday afternoon (~2h):** Item 14 (error messages), item 15 (REPL), item 12 (tension detection).

Item 13 (`--json`) is the only Phase 2 item that is a structural refactor rather than additive — save it for a dedicated session or do it last so it doesn't block the others.
