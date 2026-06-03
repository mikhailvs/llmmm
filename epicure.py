#!/usr/bin/env python3
"""
epicure - ingredient pairing & recipe ideas from food embeddings

Pipe-friendly: ingredients can come from stdin, and pair/similar/steer
output plain names when stdout is not a terminal, so commands chain:

  echo "chicken garlic lemon" | python epicure.py pair | python epicure.py recipe
  python epicure.py pair salmon miso | python epicure.py steer japanese | python epicure.py recipe
"""

import sys, os, csv, re
import numpy as np

EMBEDDING_FILE = os.path.join(os.path.dirname(__file__), "data", "epicure_cooc.csv")
PIPE_IN  = not sys.stdin.isatty()
PIPE_OUT = not sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(path=EMBEDDING_FILE):
    with open(path) as f:
        rows = list(csv.reader(f))
    names = [r[1] for r in rows[1:]]
    vecs = np.array([[float(x) for x in r[2:]] for r in rows[1:]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return names, vecs


# ---------------------------------------------------------------------------
# Stdin helpers
# ---------------------------------------------------------------------------

def read_stdin_tokens():
    """Read whitespace/newline-separated tokens from stdin."""
    return sys.stdin.read().split()


def merge_args(args):
    """Append any stdin tokens to args when stdin is a pipe."""
    if PIPE_IN:
        args = list(args) + read_stdin_tokens()
    return args


# ---------------------------------------------------------------------------
# Name resolution
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


def resolve_one(name, names, verbose=True):
    candidates = resolve(name, names)
    if not candidates:
        if verbose and not PIPE_OUT:
            print(f"  [!] '{name}' not found — skipping", file=sys.stderr)
        return None
    if candidates[0] == normalize(name):
        return candidates[0]
    if verbose and not PIPE_OUT:
        print(f"  [~] '{name}' → '{candidates[0]}'", file=sys.stderr)
    return candidates[0]


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def centroid(indices, vecs):
    v = vecs[indices].mean(axis=0)
    return v / np.linalg.norm(v)


def top_k(query_vec, vecs, names, k=12, exclude=None):
    sims = vecs @ query_vec
    if exclude:
        for i in exclude:
            sims[i] = -1.0
    order = np.argsort(-sims)
    results = []
    for i in order:
        if len(results) == k:
            break
        results.append((names[i], float(sims[i])))
    return results


def slerp(v0, v1, t):
    dot = float(np.clip(v0 @ v1, -1.0, 1.0))
    theta = np.arccos(abs(dot)) * t
    perp = v1 - dot * v0
    norm = np.linalg.norm(perp)
    if norm < 1e-8:
        return v0
    perp /= norm
    return v0 * np.cos(theta) + perp * np.sin(theta)


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
    markers = CUISINE_MARKERS.get(cuisine.lower())
    if markers is None:
        return None, f"Unknown cuisine '{cuisine}'. Choose from: {', '.join(CUISINE_MARKERS)}"
    indices = [names.index(m) for m in markers if m in names]
    if not indices:
        return None, f"No markers for '{cuisine}' found in vocabulary."
    return centroid(indices, vecs), None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_pair(args, names, vecs):
    """What goes well with these ingredients?"""
    args = merge_args(args)
    if not args:
        print("Usage: epicure.py pair <ingredient> [ingredient ...]", file=sys.stderr)
        return
    indices = [names.index(n) for a in args if (n := resolve_one(a, names))]
    if not indices:
        print("No valid ingredients found.", file=sys.stderr)
        return
    query = centroid(indices, vecs)
    results = top_k(query, vecs, names, k=15, exclude=indices)
    if PIPE_OUT:
        for name, _ in results:
            print(name)
    else:
        print(f"\nIngredients: {', '.join(names[i] for i in indices)}")
        print("─" * 50)
        print("Best complementary ingredients:")
        for name, score in results:
            print(f"  {name:<28} {score:.3f}  {'█' * int(score * 20)}")


def cmd_similar(args, names, vecs):
    """Most similar ingredients to one item."""
    args = merge_args(args)
    if not args:
        print("Usage: epicure.py similar <ingredient>", file=sys.stderr)
        return
    n = resolve_one(args[0], names)
    if not n:
        return
    idx = names.index(n)
    results = top_k(vecs[idx], vecs, names, k=15, exclude=[idx])
    if PIPE_OUT:
        for name, _ in results:
            print(name)
    else:
        print(f"\nMost similar to '{n}':")
        print("─" * 50)
        for name, score in results:
            print(f"  {name:<28} {score:.3f}  {'█' * int(score * 20)}")


def cmd_steer(args, names, vecs):
    """Steer ingredients toward a cuisine. Cuisine is always the first arg."""
    # cuisine can be in args or first token of stdin
    all_args = list(args)
    if PIPE_IN:
        all_args += read_stdin_tokens()
    if len(all_args) < 2:
        print("Usage: epicure.py steer <cuisine> <ingredient> [ingredient ...]", file=sys.stderr)
        print(f"Cuisines: {', '.join(CUISINE_MARKERS)}", file=sys.stderr)
        return
    cuisine, ingredient_args = all_args[0], all_args[1:]
    cvec, err = cuisine_vector(cuisine, names, vecs)
    if err:
        print(err, file=sys.stderr)
        return
    indices = [names.index(n) for a in ingredient_args if (n := resolve_one(a, names))]
    if not indices:
        print("No valid ingredients found.", file=sys.stderr)
        return
    base = centroid(indices, vecs)
    if PIPE_OUT:
        # emit medium-strength suggestions, one per line
        steered = slerp(base, cvec, 0.5)
        for name, _ in top_k(steered, vecs, names, k=10, exclude=indices):
            print(name)
    else:
        print(f"\nBase: {', '.join(names[i] for i in indices)}")
        print(f"Steering toward: {cuisine}")
        print("─" * 50)
        for t, label in [(0.25, "gentle"), (0.5, "medium"), (0.75, "strong")]:
            steered = slerp(base, cvec, t)
            results = top_k(steered, vecs, names, k=6, exclude=indices)
            print(f"  [{label:6}] {', '.join(r[0] for r in results)}")


def cmd_recipe(args, names, vecs):
    """Generate a recipe via local Ollama."""
    import urllib.request, json

    OLLAMA_URL = "http://localhost:11434/api/chat"
    MODEL = os.environ.get("EPICURE_MODEL", "qwen3:8b")

    args = merge_args(args)
    if not args:
        print("Usage: epicure.py recipe <ingredient> [ingredient ...]", file=sys.stderr)
        return
    indices = [names.index(n) for a in args if (n := resolve_one(a, names))]
    if not indices:
        print("No valid ingredients found.", file=sys.stderr)
        return
    base_ingredients = [names[i] for i in indices]
    query = centroid(indices, vecs)
    complements = top_k(query, vecs, names, k=8, exclude=indices)
    complement_names = [c[0] for c in complements]

    print(f"\nCore: {', '.join(base_ingredients)}", file=sys.stderr)
    print(f"Pairings: {', '.join(complement_names)}", file=sys.stderr)
    print(f"Generating with {MODEL}...\n", file=sys.stderr)

    prompt = (
        f"You are a skilled chef. I have these ingredients: {', '.join(base_ingredients)}.\n"
        f"Based on flavor science, these pair especially well with them: {', '.join(complement_names)}.\n\n"
        f"Create one delicious, practical recipe using the core ingredients and at least 3 of the suggested additions. "
        f"Format it as:\n"
        f"# [Recipe Name]\n\n"
        f"**Serves:** N\n**Time:** X min\n\n"
        f"## Ingredients\n(bullet list)\n\n"
        f"## Instructions\n(numbered steps)\n\n"
        f"Keep it concise and achievable for a home cook. "
        f"Skip any preamble — output the recipe directly."
    )

    payload = json.dumps({
        "model": MODEL,
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
        print(f"\n[!] Could not reach Ollama at {OLLAMA_URL}: {e}", file=sys.stderr)
        print("    Start it with: ollama serve", file=sys.stderr)
    print()


def cmd_search(args, names, _vecs):
    """Search the ingredient vocabulary."""
    args = merge_args(args)
    if not args:
        print("Usage: epicure.py search <term>", file=sys.stderr)
        return
    term = normalize(args[0])
    matches = [n for n in names if term in n]
    if matches:
        if not PIPE_OUT:
            print(f"\nIngredients matching '{args[0]}':")
        for m in matches:
            print(f"  {m}" if not PIPE_OUT else m)
    else:
        print(f"No ingredients matching '{args[0]}'.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "pair":    cmd_pair,
    "similar": cmd_similar,
    "steer":   cmd_steer,
    "recipe":  cmd_recipe,
    "search":  cmd_search,
}

HELP = """\
epicure — offline ingredient pairing & recipe tool

Commands:
  pair    <ingredients…>           what goes well with these ingredients?
  similar <ingredient>             most similar ingredients
  steer   <cuisine> <ingredients…> what to add to push toward a cuisine
  recipe  <ingredients…>           generate a recipe (via local Ollama)
  search  <term>                   search the 1,790-ingredient vocabulary

Cuisines: italian, mexican, japanese, indian, chinese, thai,
          french, mediterranean, korean, middle_eastern

Pipe-friendly — ingredients can come from stdin, and pair/similar/steer
emit plain names when stdout is not a terminal:

  echo "chicken garlic lemon"  | python epicure.py pair
  echo "chicken garlic lemon"  | python epicure.py pair   | python epicure.py recipe
  echo "salmon miso"           | python epicure.py steer japanese | python epicure.py recipe
  python epicure.py pair beef onion | python epicure.py recipe

Override the Ollama model:
  EPICURE_MODEL=llama3.2 python epicure.py recipe chicken garlic
"""

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'. Run with --help for usage.", file=sys.stderr)
        sys.exit(1)

    if not PIPE_OUT:
        print("Loading embeddings...", end=" ", flush=True)
    names, vecs = load_embeddings()
    if not PIPE_OUT:
        print(f"({len(names)} ingredients)")

    COMMANDS[cmd](sys.argv[2:], names, vecs)
