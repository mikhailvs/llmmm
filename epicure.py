#!/usr/bin/env python3
"""
epicure - ingredient pairing & recipe ideas from food embeddings

Pipe-friendly: ingredients can come from stdin, and pair/similar/steer
output plain names when stdout is not a terminal, so commands chain:

  epicure pair salmon miso | epicure steer japanese | epicure recipe
  epicure pair chicken garlic lemon | epicure recipe
"""

import sys, os, csv, re
import numpy as np

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
FILE_COOC    = os.path.join(DATA_DIR, "epicure_cooc.csv")
FILE_CHEM    = os.path.join(DATA_DIR, "epicure_chem.csv")
FILE_CORE    = os.path.join(DATA_DIR, "epicure_core.csv")
PIPE_IN      = not sys.stdin.isatty()
PIPE_OUT     = not sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display(name):
    """Convert internal underscore name to human-readable form."""
    return name.replace("_", " ")


def display_list(names):
    return ", ".join(display(n) for n in names)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_one(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    names = [r[1] for r in rows[1:]]
    vecs  = np.array([[float(x) for x in r[2:]] for r in rows[1:]], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.where(norms == 0, 1.0, norms)
    return names, vecs


_CACHE = {}

def load_embeddings(variant="cooc"):
    if variant in _CACHE:
        return _CACHE[variant]
    path = {"cooc": FILE_COOC, "chem": FILE_CHEM, "core": FILE_CORE}[variant]
    names, vecs = _load_one(path)
    # Precompute per-ingredient mean similarity for hubness correction
    # mean_sim[i] = average cosine similarity of ingredient i to all others
    mean_sim = (vecs @ vecs.T).mean(axis=1)
    _CACHE[variant] = (names, vecs, mean_sim)
    return names, vecs, mean_sim


def load_all_embeddings():
    """Load all three variants, return (names, cooc_vecs, chem_vecs, core_vecs, mean_sims)."""
    names_c, cooc, ms_cooc = load_embeddings("cooc")
    names_h, chem, ms_chem = load_embeddings("chem")
    names_r, core, ms_core = load_embeddings("core")
    # All three files share the same ingredient list in the same order
    return names_c, cooc, chem, core, ms_cooc, ms_chem, ms_core


# ---------------------------------------------------------------------------
# Stdin helpers
# ---------------------------------------------------------------------------

def read_stdin_tokens():
    return sys.stdin.read().split()


def merge_args(args):
    if PIPE_IN:
        args = list(args) + read_stdin_tokens()
    return args


# ---------------------------------------------------------------------------
# Name resolution with fuzzy suggestions
# ---------------------------------------------------------------------------

def normalize(s):
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def resolve(name, names):
    key = normalize(name)
    if key in names:
        return [key]
    matches = [n for n in names if key in n or n.startswith(key)]
    if matches:
        return matches
    words = set(key.split("_"))
    scored = [(len(words & set(n.split("_"))), n) for n in names]
    best = max(s for s, _ in scored)
    if best > 0:
        return [n for s, n in scored if s == best][:5]
    return []


def fuzzy_suggest(key, names, vecs, n=3):
    """Use embedding similarity to suggest near-misses for unknown ingredients."""
    # Simple character-overlap heuristic (no embedding needed since we can't look it up)
    key_chars = set(key)
    scored = [(len(key_chars & set(n)) / max(len(key_chars | set(n)), 1), n)
              for n in names if abs(len(n) - len(key)) <= 4]
    scored.sort(reverse=True)
    return [n for _, n in scored[:n]]


def resolve_one(name, names, vecs=None, quiet=False):
    candidates = resolve(name, names)
    if not candidates:
        if not quiet:
            suggestions = fuzzy_suggest(normalize(name), names, vecs) if vecs is not None else []
            msg = f"  '{display(normalize(name))}' not found"
            if suggestions:
                msg += f" — did you mean: {display_list(suggestions[:3])}?"
            else:
                msg += " — try: epicure search <term>"
            print(msg, file=sys.stderr)
        return None
    if candidates[0] == normalize(name):
        return candidates[0]
    if not quiet and not PIPE_OUT:
        print(f"  '{display(normalize(name))}' -> '{display(candidates[0])}'", file=sys.stderr)
    return candidates[0]


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def idf_centroid(indices, vecs, names_list):
    """IDF-weighted centroid: weight each ingredient vector by log(N / df).
    df is approximated by name length as a proxy for specificity; for a true
    IDF we use the document frequency implied by co-occurrence rank (index+1).
    In practice we weight by log(N / (rank+1)) where rank is the vocab index,
    since the CSV is ordered by frequency in the source corpus."""
    N = len(names_list)
    weights = np.array([np.log(N / (idx + 1)) for idx in indices], dtype=np.float32)
    weights = np.maximum(weights, 0.1)
    weighted = (vecs[indices] * weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(weighted)
    if norm < 1e-8:
        return vecs[indices].mean(axis=0)
    return weighted / norm


def top_k(query_vec, vecs, names, k=12, exclude=None, mean_sim=None):
    sims = vecs @ query_vec
    if mean_sim is not None:
        sims = sims - mean_sim          # hubness correction
    if exclude:
        for i in exclude:
            sims[i] = -np.inf
    order = np.argsort(-sims)
    results = []
    for i in order:
        if len(results) == k:
            break
        results.append((names[i], float(sims[i])))
    return results


def mmr(query_vec, vecs, names, k=12, exclude=None, mean_sim=None, lam=0.6):
    """Maximal Marginal Relevance: balance relevance vs. diversity."""
    # Get a larger candidate pool first
    pool_size = min(k * 6, len(names))
    candidates = top_k(query_vec, vecs, names, k=pool_size,
                        exclude=exclude, mean_sim=mean_sim)
    if not candidates:
        return []

    cand_names  = [c[0] for c in candidates]
    cand_scores = np.array([c[1] for c in candidates], dtype=np.float32)
    # Normalize scores to [0,1] for the MMR formula
    s_min, s_max = cand_scores.min(), cand_scores.max()
    if s_max > s_min:
        cand_scores = (cand_scores - s_min) / (s_max - s_min)

    name_to_idx = {n: names.index(n) for n in cand_names}
    selected = []
    selected_vecs = []

    for _ in range(k):
        if not cand_names:
            break
        if not selected_vecs:
            # First pick: best relevance
            best_idx = int(np.argmax(cand_scores))
        else:
            sel_mat   = np.array(selected_vecs, dtype=np.float32)
            redundancy = np.array([
                (vecs[name_to_idx[n]] @ sel_mat.T).max()
                for n in cand_names
            ], dtype=np.float32)
            mmr_scores = lam * cand_scores - (1 - lam) * redundancy
            best_idx   = int(np.argmax(mmr_scores))

        chosen = cand_names[best_idx]
        chosen_score = float(candidates[best_idx][1])
        selected.append((chosen, chosen_score))
        selected_vecs.append(vecs[name_to_idx[chosen]])
        cand_names  = [n for i, n in enumerate(cand_names)  if i != best_idx]
        candidates  = [c for i, c in enumerate(candidates)   if i != best_idx]
        cand_scores = np.delete(cand_scores, best_idx)

    return selected


def slerp(v0, v1, t):
    """Spherical linear interpolation. No abs() — preserves hemisphere."""
    dot = float(np.clip(v0 @ v1, -1.0, 1.0))
    theta = np.arccos(dot) * t
    perp = v1 - dot * v0
    norm = np.linalg.norm(perp)
    if norm < 1e-8:
        return v0
    perp /= norm
    return v0 * np.cos(theta) + perp * np.sin(theta)


def ensemble_score(query_vec_cooc, query_vec_chem, query_vec_core,
                   cooc, chem, core, names,
                   ms_cooc, ms_chem, ms_core,
                   k=12, exclude=None,
                   w_cooc=0.4, w_chem=0.4, w_core=0.2):
    """Weighted ensemble across all three embedding spaces."""
    def adjusted(vecs, qvec, ms):
        s = vecs @ qvec
        if ms is not None:
            s = s - ms
        return s

    s_cooc = adjusted(cooc, query_vec_cooc, ms_cooc)
    s_chem = adjusted(chem, query_vec_chem, ms_chem)
    s_core = adjusted(core, query_vec_core, ms_core)

    # Z-normalise each before combining so different scales don't dominate
    def znorm(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / (sd + 1e-8)

    combined = w_cooc * znorm(s_cooc) + w_chem * znorm(s_chem) + w_core * znorm(s_core)

    if exclude:
        for i in exclude:
            combined[i] = -np.inf

    order = np.argsort(-combined)
    results = []
    for i in order:
        if len(results) == k:
            break
        results.append((names[i], float(combined[i])))
    return results


# ---------------------------------------------------------------------------
# Cuisine poles
# ---------------------------------------------------------------------------

CUISINE_MARKERS = {
    "italian":        ["basil", "tomato", "olive_oil", "oregano", "pasta",
                       "parmesan_cheese", "mozzarella_cheese", "thyme", "rosemary"],
    "mexican":        ["cumin", "chili_pepper", "lime", "avocado", "coriander",
                       "jalapeno", "tortilla", "corn", "chipotle"],
    "japanese":       ["soy_sauce", "mirin", "miso", "ginger", "sesame_oil",
                       "sake", "nori", "dashi", "wasabi"],
    "indian":         ["turmeric", "cumin", "coriander", "garam_masala",
                       "cardamom", "chili_pepper", "ginger", "ghee", "fenugreek"],
    "chinese":        ["soy_sauce", "ginger", "sesame_oil", "oyster_sauce",
                       "hoisin_sauce", "star_anise", "five_spice_powder", "doubanjiang"],
    "thai":           ["fish_sauce", "lemongrass", "galangal", "coconut_milk",
                       "chili_pepper", "basil", "lime", "shrimp_paste"],
    "french":         ["butter", "cream", "thyme", "tarragon", "shallot",
                       "dijon_mustard", "wine", "bay_leaf", "leek"],
    "mediterranean":  ["olive_oil", "lemon", "garlic", "oregano", "thyme",
                       "feta_cheese", "tomato", "eggplant", "chickpea"],
    "korean":         ["gochujang", "sesame_oil", "soy_sauce", "ginger",
                       "garlic", "rice_vinegar", "sesame_seed", "doenjang"],
    "middle_eastern": ["cumin", "coriander", "turmeric", "sumac", "za_atar",
                       "tahini", "pomegranate_molasses", "chickpea", "lemon"],
}


def cuisine_vector(cuisine, names, vecs):
    """Return a discriminative cuisine pole with the global mean subtracted."""
    markers = CUISINE_MARKERS.get(cuisine.lower())
    if markers is None:
        return None, f"Unknown cuisine '{cuisine}'. Choose from: {', '.join(CUISINE_MARKERS)}"
    indices = [names.index(m) for m in markers if m in names]
    if not indices:
        return None, f"No markers for '{cuisine}' found in vocabulary."
    # Subtract global corpus mean to isolate cuisine-specific direction
    global_mean = vecs.mean(axis=0)
    marker_vecs = vecs[indices] - global_mean
    v = marker_vecs.mean(axis=0)
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return None, f"Cuisine pole for '{cuisine}' collapsed to zero after mean subtraction."
    return v / norm, None


# ---------------------------------------------------------------------------
# Surprise / chemistry gap
# ---------------------------------------------------------------------------

def surprise_scores(query_indices, cooc, chem, names, ms_cooc, ms_chem, k=12, exclude=None):
    """Surface scientifically compatible but culturally unusual pairings.

    Computes chem_z - cooc_z: high score = good chemistry, unusual co-occurrence.
    """
    from numpy import array, argsort

    qvec_cooc = idf_centroid(query_indices, cooc, names)
    qvec_chem = idf_centroid(query_indices, chem, names)

    s_cooc = cooc @ qvec_cooc - ms_cooc
    s_chem = chem @ qvec_chem - ms_chem

    def znorm(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / (sd + 1e-8)

    gap = znorm(s_chem) - znorm(s_cooc)

    if exclude:
        for i in exclude:
            gap[i] = -np.inf

    order = argsort(-gap)
    results = []
    for i in order:
        if len(results) == k:
            break
        results.append((names[i], float(gap[i])))
    return results


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def pop_flag(args, flag):
    """Remove a boolean flag from args list and return (found, remaining_args)."""
    if flag in args:
        args = [a for a in args if a != flag]
        return True, args
    return False, args


def pop_option(args, flag):
    """Remove --flag VALUE from args and return (value_or_None, remaining_args)."""
    try:
        i = args.index(flag)
        val = args[i + 1]
        args = args[:i] + args[i + 2:]
        return val, args
    except (ValueError, IndexError):
        return None, args


# ---------------------------------------------------------------------------
# Loading message
# ---------------------------------------------------------------------------

def loading_msg(msg, quiet=False):
    if not quiet and sys.stderr.isatty():
        print(msg, end=" ", flush=True, file=sys.stderr)


def loading_done(msg, quiet=False):
    if not quiet and sys.stderr.isatty():
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_pair(args, quiet=False):
    """What goes well with these ingredients?

    --variant {cooc,chem,core,ensemble}   default: chem (finds molecular compatibility)
    --creative                            highlight surprising chemistry-vs-culture gaps
    """
    variant, args = pop_option(args, "--variant")
    creative, args = pop_flag(args, "--creative")
    args = merge_args(args)
    if not args:
        print("Usage: epicure pair [--variant cooc|chem|core|ensemble] [--creative] <ingredient> ...",
              file=sys.stderr)
        return 1

    if creative:
        # surprise mode: load both cooc and chem
        loading_msg("Loading embeddings...", quiet)
        names, cooc, chem, core, ms_cooc, ms_chem, ms_core = load_all_embeddings()
        loading_done(f"({len(names)} ingredients)", quiet)
        indices = [names.index(n) for a in args if (n := resolve_one(a, names, cooc))]
        if not indices:
            print("No valid ingredients found.", file=sys.stderr)
            return 1
        results = surprise_scores(indices, cooc, chem, names, ms_cooc, ms_chem, k=15, exclude=indices)
        label = "Surprising pairings (good chemistry, unusual in recipes)"
    else:
        # default for pair: chem (molecular compatibility)
        v = variant or "chem"
        if v == "ensemble":
            loading_msg("Loading embeddings...", quiet)
            names, cooc, chem, core, ms_cooc, ms_chem, ms_core = load_all_embeddings()
            loading_done(f"({len(names)} ingredients)", quiet)
            indices = [names.index(n) for a in args if (n := resolve_one(a, names, cooc))]
            if not indices:
                print("No valid ingredients found.", file=sys.stderr)
                return 1
            qcooc = idf_centroid(indices, cooc, names)
            qchem = idf_centroid(indices, chem, names)
            qcore = idf_centroid(indices, core, names)
            results = ensemble_score(qcooc, qchem, qcore, cooc, chem, core, names,
                                     ms_cooc, ms_chem, ms_core, k=15, exclude=indices)
        else:
            if v not in ("cooc", "chem", "core"):
                print(f"Unknown variant '{v}'. Choose: cooc, chem, core, ensemble", file=sys.stderr)
                return 1
            loading_msg("Loading embeddings...", quiet)
            names, vecs, mean_sim = load_embeddings(v)
            loading_done(f"({len(names)} ingredients)", quiet)
            indices = [names.index(n) for a in args if (n := resolve_one(a, names, vecs))]
            if not indices:
                print("No valid ingredients found.", file=sys.stderr)
                return 1
            query = idf_centroid(indices, vecs, names)
            results = mmr(query, vecs, names, k=15, exclude=indices, mean_sim=mean_sim)
        label = "Best complementary ingredients"

    if PIPE_OUT:
        for name, _ in results:
            print(name)
    else:
        print(f"\nIngredients: {display_list(names[i] for i in indices)}")
        print("-" * 50)
        print(f"{label}:")
        for name, score in results:
            bar = int(max(score, 0) * 20)
            print(f"  {display(name):<30} {score:+.3f}  {'|' * bar}")
    return 0


def cmd_similar(args, quiet=False):
    """Most similar ingredients to one item.

    --variant {cooc,chem,core}   default: cooc (cultural substitutes)
    """
    variant, args = pop_option(args, "--variant")
    args = merge_args(args)
    if not args:
        print("Usage: epicure similar [--variant cooc|chem|core] <ingredient>", file=sys.stderr)
        return 1

    v = variant or "cooc"
    if v not in ("cooc", "chem", "core"):
        print(f"Unknown variant '{v}'.", file=sys.stderr)
        return 1

    loading_msg("Loading embeddings...", quiet)
    names, vecs, mean_sim = load_embeddings(v)
    loading_done(f"({len(names)} ingredients)", quiet)

    n = resolve_one(args[0], names, vecs)
    if not n:
        return 1
    idx = names.index(n)
    results = mmr(vecs[idx], vecs, names, k=15, exclude=[idx], mean_sim=mean_sim)

    if PIPE_OUT:
        for name, _ in results:
            print(name)
    else:
        print(f"\nMost similar to '{display(n)}':")
        print("-" * 50)
        for name, score in results:
            bar = int(max(score, 0) * 20)
            print(f"  {display(name):<30} {score:+.3f}  {'|' * bar}")
    return 0


def cmd_steer(args, quiet=False):
    """Steer ingredients toward a cuisine.

    Usage: epicure steer <cuisine> <ingredient> [ingredient ...]
           epicure steer --from <cuisine> --to <cuisine> <ingredient> ...

    --variant {cooc,core}   default: cooc (cultural grammar)
    --from <cuisine>        compute relative direction between two cuisine poles
    --to   <cuisine>        (used together with --from)
    """
    variant,   args = pop_option(args, "--variant")
    from_cui,  args = pop_option(args, "--from")
    to_cui,    args = pop_option(args, "--to")

    all_args = list(args)
    if PIPE_IN:
        all_args += read_stdin_tokens()

    v = variant or "cooc"
    if v not in ("cooc", "core"):
        print(f"Unknown variant '{v}'. Choose: cooc, core", file=sys.stderr)
        return 1

    loading_msg("Loading embeddings...", quiet)
    names, vecs, mean_sim = load_embeddings(v)
    loading_done(f"({len(names)} ingredients)", quiet)

    if from_cui and to_cui:
        # Analogy arithmetic: direction = to_pole - from_pole
        pole_from, err = cuisine_vector(from_cui, names, vecs)
        if err:
            print(err, file=sys.stderr); return 1
        pole_to, err = cuisine_vector(to_cui, names, vecs)
        if err:
            print(err, file=sys.stderr); return 1
        direction = pole_to - pole_from
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            print("From and to cuisines are too similar.", file=sys.stderr); return 1
        direction /= norm
        cuisine_label = f"{from_cui} -> {to_cui}"
        ingredient_args = all_args
    else:
        if len(all_args) < 2:
            print("Usage: epicure steer <cuisine> <ingredient> [ingredient ...]",
                  file=sys.stderr)
            print(f"Cuisines: {', '.join(CUISINE_MARKERS)}", file=sys.stderr)
            return 1
        cuisine_label    = all_args[0]
        ingredient_args  = all_args[1:]
        direction, err = cuisine_vector(cuisine_label, names, vecs)
        if err:
            print(err, file=sys.stderr); return 1

    indices = [names.index(n) for a in ingredient_args if (n := resolve_one(a, names, vecs))]
    if not indices:
        print("No valid ingredients found.", file=sys.stderr)
        return 1

    base = idf_centroid(indices, vecs, names)

    if PIPE_OUT:
        steered = slerp(base, direction, 0.5)
        for name, _ in top_k(steered, vecs, names, k=10, exclude=indices, mean_sim=mean_sim):
            print(name)
    else:
        base_display = display_list(names[i] for i in indices)
        print(f"\nBase: {base_display} -> {cuisine_label}")
        print("-" * 50)
        for t, label in [(0.25, "gentle"), (0.5, "medium"), (0.75, "strong")]:
            steered = slerp(base, direction, t)
            results = top_k(steered, vecs, names, k=6, exclude=indices, mean_sim=mean_sim)
            names_str = " · ".join(display(r[0]) for r in results)
            print(f"  {label:6}  {names_str}")
    return 0


def _check_ollama(model, quiet=False):
    """Return True if Ollama is reachable, else print instructions and return False."""
    import socket
    try:
        sock = socket.create_connection(("localhost", 11434), timeout=2)
        sock.close()
        return True
    except OSError:
        print("\nOllama is not running.", file=sys.stderr)
        print("Start it with:  ollama serve", file=sys.stderr)
        print("Install from:   https://ollama.com", file=sys.stderr)
        print(f"Then pull a model:  ollama pull {model}", file=sys.stderr)
        return False


def _stream_recipe(prompt, model, quiet=False):
    """Stream a recipe from Ollama, return exit code."""
    import urllib.request, json
    OLLAMA_URL = "http://localhost:11434/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": False,
        "options": {"num_predict": 1024},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                chunk = json.loads(line)
                print(chunk.get("message", {}).get("content", ""), end="", flush=True)
                if chunk.get("done"):
                    break
    except OSError as e:
        print(f"\nCould not reach Ollama: {e}", file=sys.stderr)
        return 1
    print()
    return 0


def _dietary_note(args):
    """Pop dietary flags and return (constraint_string, remaining_args)."""
    constraints = []
    for flag, label in [("--vegan", "vegan (no meat, fish, dairy, or eggs)"),
                         ("--vegetarian", "vegetarian (no meat or fish)"),
                         ("--gluten-free", "gluten-free"),
                         ("--dairy-free", "dairy-free (no milk, cream, butter, or cheese)")]:
        found, args = pop_flag(args, flag)
        if found:
            constraints.append(label)
    return constraints, args


def cmd_recipe(args, quiet=False):
    """Generate a recipe via local Ollama.

    --model <name>      Ollama model (default: $EPICURE_MODEL or qwen3:8b)
    --serves <N>        target number of servings
    --time <N>          maximum cook time in minutes
    --vegan             dietary constraint
    --vegetarian        dietary constraint
    --gluten-free       dietary constraint
    --dairy-free        dietary constraint
    """
    model_arg,  args = pop_option(args, "--model")
    serves,     args = pop_option(args, "--serves")
    max_time,   args = pop_option(args, "--time")
    constraints, args = _dietary_note(args)
    args = merge_args(args)

    MODEL = model_arg or os.environ.get("EPICURE_MODEL", "qwen3:8b")

    if not args:
        print("Usage: epicure recipe [--model M] [--serves N] [--time N] "
              "[--vegan] [--gluten-free] [--dairy-free] <ingredient> ...", file=sys.stderr)
        return 1

    if not _check_ollama(MODEL):
        return 1

    loading_msg("Loading embeddings...", quiet)
    names, cooc, chem, core, ms_cooc, ms_chem, ms_core = load_all_embeddings()
    loading_done(f"({len(names)} ingredients)", quiet)

    indices = [names.index(n) for a in args if (n := resolve_one(a, names, cooc))]
    if not indices:
        print("No valid ingredients found.", file=sys.stderr)
        return 1

    base_ingredients = [names[i] for i in indices]
    qcooc = idf_centroid(indices, cooc, names)
    qchem = idf_centroid(indices, chem, names)
    qcore = idf_centroid(indices, core, names)
    complement_results = ensemble_score(
        qcooc, qchem, qcore, cooc, chem, core, names,
        ms_cooc, ms_chem, ms_core, k=8, exclude=indices
    )
    complement_names = [c[0] for c in complement_results]

    print(f"\nCore: {display_list(base_ingredients)}", file=sys.stderr)
    print(f"Pairings: {display_list(complement_names)}", file=sys.stderr)
    print(f"Generating with {MODEL}...\n", file=sys.stderr)

    serves_line  = f"**Serves:** {serves}" if serves else "**Serves:** 2–4"
    time_line    = f"**Time:** {max_time} min or less" if max_time else "**Time:** X min"
    diet_line    = (f"Dietary requirements: {', '.join(constraints)}. " if constraints else "")
    time_constraint = (f"The recipe must be completable in {max_time} minutes. " if max_time else "")

    prompt = (
        f"You are a skilled chef. I have these ingredients: {display_list(base_ingredients)}.\n"
        f"Based on flavor science, these pair especially well with them: {display_list(complement_names)}.\n\n"
        f"{diet_line}{time_constraint}"
        f"Create one delicious, practical recipe using the core ingredients and at least 3 of the "
        f"suggested additions. Format it as:\n"
        f"# [Recipe Name]\n\n"
        f"{serves_line}\n{time_line}\n\n"
        f"## Ingredients\n(bullet list)\n\n"
        f"## Instructions\n(numbered steps)\n\n"
        f"Keep it concise and achievable for a home cook. Skip any preamble — output the recipe directly."
    )
    return _stream_recipe(prompt, MODEL, quiet)


def cmd_fridge(args, quiet=False):
    """Make dinner from exactly what you have — no shopping required.

    Unlike 'recipe', this command frames the ingredients as your full
    available pantry and tells the LLM not to require anything else.

    --model <name>      Ollama model
    --serves <N>        target servings
    --time <N>          max cook time in minutes
    --vegan / --vegetarian / --gluten-free / --dairy-free
    """
    model_arg,  args = pop_option(args, "--model")
    serves,     args = pop_option(args, "--serves")
    max_time,   args = pop_option(args, "--time")
    constraints, args = _dietary_note(args)
    args = merge_args(args)

    MODEL = model_arg or os.environ.get("EPICURE_MODEL", "qwen3:8b")

    if not args:
        print("Usage: epicure fridge [--serves N] [--time N] [--vegan] <ingredient> ...",
              file=sys.stderr)
        print("Example: epicure fridge chicken lemon garlic spinach", file=sys.stderr)
        return 1

    if not _check_ollama(MODEL):
        return 1

    loading_msg("Loading embeddings...", quiet)
    names, cooc, chem, core, ms_cooc, ms_chem, ms_core = load_all_embeddings()
    loading_done(f"({len(names)} ingredients)", quiet)

    indices = [names.index(n) for a in args if (n := resolve_one(a, names, cooc))]
    if not indices:
        print("No valid ingredients found.", file=sys.stderr)
        return 1

    have = [names[i] for i in indices]

    # Find which of their ingredients pair best together (to guide the LLM)
    qcooc = idf_centroid(indices, cooc, names)
    qchem = idf_centroid(indices, chem, names)
    qcore = idf_centroid(indices, core, names)
    pair_results = ensemble_score(
        qcooc, qchem, qcore, cooc, chem, core, names,
        ms_cooc, ms_chem, ms_core, k=5, exclude=indices
    )
    # Only keep pairings that are also in the user's ingredient list
    have_set = set(have)
    good_combos = [display(have[i]) for i in range(len(have))
                   if any(r[0] in have_set for r in pair_results)]

    print(f"\nFridge: {display_list(have)}", file=sys.stderr)
    print(f"Generating with {MODEL}...\n", file=sys.stderr)

    serves_line  = f"**Serves:** {serves}" if serves else "**Serves:** 2–4"
    time_line    = f"**Time:** {max_time} min or less" if max_time else "**Time:** X min"
    diet_line    = (f"Dietary requirements: {', '.join(constraints)}. " if constraints else "")
    time_constraint = (f"The recipe must be completable in {max_time} minutes. " if max_time else "")

    prompt = (
        f"You are a skilled chef helping someone cook dinner from what they already have at home.\n"
        f"They have ONLY these ingredients available: {display_list(have)}.\n"
        f"They cannot go shopping. Build the entire recipe from this list only — "
        f"you may assume they have basic pantry staples (salt, pepper, oil, water) "
        f"but nothing else.\n\n"
        f"{diet_line}{time_constraint}"
        f"Create the best possible recipe from these ingredients. Format it as:\n"
        f"# [Recipe Name]\n\n"
        f"{serves_line}\n{time_line}\n\n"
        f"## Ingredients\n(bullet list, only from the available list)\n\n"
        f"## Instructions\n(numbered steps)\n\n"
        f"Skip any preamble — output the recipe directly."
    )
    return _stream_recipe(prompt, MODEL, quiet)


def cmd_search(args, quiet=False):
    """Search the ingredient vocabulary."""
    args = merge_args(args)
    if not args:
        print("Usage: epicure search <term>", file=sys.stderr)
        return 1
    term = normalize(args[0])
    # load just cooc names (fast, no vecs needed for search)
    names, _vecs, _ms = load_embeddings("cooc")
    matches = [n for n in names if term in n]
    if matches:
        if not PIPE_OUT:
            print(f"\nIngredients matching '{args[0]}':")
        for m in matches:
            print(f"  {display(m)}" if not PIPE_OUT else m)
    else:
        print(f"No ingredients matching '{display(term)}'.", file=sys.stderr)
        return 1
    return 0


def cmd_surprise(args, quiet=False):
    """Find scientifically compatible but culturally unusual pairings.

    These are ingredients with high molecular compatibility (chem) but
    low cultural co-occurrence (cooc): surprising creative pairings.

    Usage: epicure surprise <ingredient> [ingredient ...]
    """
    args = merge_args(args)
    if not args:
        print("Usage: epicure surprise <ingredient> [ingredient ...]", file=sys.stderr)
        return 1

    loading_msg("Loading embeddings...", quiet)
    names, cooc, chem, core, ms_cooc, ms_chem, ms_core = load_all_embeddings()
    loading_done(f"({len(names)} ingredients)", quiet)

    indices = [names.index(n) for a in args if (n := resolve_one(a, names, cooc))]
    if not indices:
        print("No valid ingredients found.", file=sys.stderr)
        return 1

    results = surprise_scores(indices, cooc, chem, names, ms_cooc, ms_chem, k=15, exclude=indices)

    if PIPE_OUT:
        for name, _ in results:
            print(name)
    else:
        print(f"\nSurprising pairings for: {display_list(names[i] for i in indices)}")
        print("(high molecular compatibility, unusual in recipes)")
        print("-" * 50)
        for name, score in results:
            bar = int(max(score * 3, 0))
            print(f"  {display(name):<30} {score:+.3f}  {'|' * min(bar, 20)}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "pair":     cmd_pair,
    "similar":  cmd_similar,
    "steer":    cmd_steer,
    "recipe":   cmd_recipe,
    "fridge":   cmd_fridge,
    "search":   cmd_search,
    "find":     cmd_search,      # alias
    "surprise": cmd_surprise,
}

HELP = """\
epicure — offline ingredient pairing & recipe tool

Commands:
  search   <term>                        search the 1,790-ingredient vocabulary
  find     <term>                        alias for search
  pair     [--variant V] <ingredients…>  what goes well with these?
  similar  [--variant V] <ingredient>    most similar ingredients
  steer    <cuisine> <ingredients…>      push ingredients toward a cuisine style
  steer    --from <c1> --to <c2> <ingr>  relative direction between two cuisines
  surprise <ingredients…>               surprising: good chemistry, rarely combined
  fridge   <ingredients…>               make dinner from exactly what you have
  recipe   <ingredients…>               generate a recipe (suggests additions)

Variants (--variant):
  cooc      cultural co-occurrence (default for similar, steer)
  chem      molecular flavor compatibility (default for pair)
  core      cross-lingual core embeddings
  ensemble  40% cooc + 40% chem + 20% core (pair, recipe)

Cuisines: italian · mexican · japanese · indian · chinese · thai
          french · mediterranean · korean · middle_eastern

Recipe / fridge flags:
  --serves <N>        target number of servings
  --time <N>          max cook time in minutes
  --vegan             dietary constraint
  --vegetarian        dietary constraint
  --gluten-free       dietary constraint
  --dairy-free        dietary constraint
  --model <name>      Ollama model (default: $EPICURE_MODEL or qwen3:8b)

Global flags:
  -q / --quiet    suppress loading messages
  --creative      pair mode: chemistry-vs-culture gap (surprising pairings)
  --from/--to     steer mode: relative direction between two cuisine poles

Pipe-friendly — one ingredient per line when stdout is not a terminal:
  epicure pair salmon miso | epicure steer japanese | epicure recipe
  epicure pair chicken garlic lemon | epicure recipe
  epicure surprise chocolate | epicure recipe

Examples:
  epicure fridge chicken lemon garlic spinach --serves 2
  epicure fridge salmon miso scallion --time 20 --gluten-free
  epicure recipe beef onion --serves 4 --time 45
  epicure steer --from italian --to japanese salmon garlic
  epicure surprise chocolate
  epicure search pepper
"""

def main():
    quiet, sys.argv = pop_flag(sys.argv, "-q")
    quiet2, sys.argv = pop_flag(sys.argv, "--quiet")
    quiet = quiet or quiet2

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd not in COMMANDS:
        # Suggest search on unknown command (might be an ingredient name)
        try:
            names, _vecs, _ = load_embeddings("cooc")
            suggestions = fuzzy_suggest(normalize(cmd), names, _vecs, n=3)
            msg = f"Unknown command '{cmd}'."
            if suggestions:
                msg += f" Did you mean: epicure search {cmd}?"
            msg += " Run epicure --help for usage."
            print(msg, file=sys.stderr)
        except Exception:
            print(f"Unknown command '{cmd}'. Run epicure --help for usage.", file=sys.stderr)
        sys.exit(1)

    rc = COMMANDS[cmd](sys.argv[2:], quiet=quiet)
    sys.exit(rc if rc else 0)


if __name__ == "__main__":
    main()
