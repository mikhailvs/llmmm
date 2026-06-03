# epicure

Offline ingredient pairing and recipe generation, powered by food embeddings from the [Epicure paper](https://arxiv.org/abs/2605.22391).

Three embedding spaces encode complementary knowledge learned from 4.1 million recipes: cultural co-occurrence (`cooc`), molecular flavor compatibility (`chem`), and cross-lingual core structure (`core`). The tool finds ingredient pairings, steers toward cuisine styles, surfaces surprising creative pairings, and generates recipes with a local LLM via Ollama — no internet required after setup.

## Setup

**Requirements:** Python 3.10+, numpy, [Ollama](https://ollama.com) (for recipe generation)

```bash
pip install -e .          # installs the 'epicure' command
ollama pull qwen3:8b
```

Download the three embedding files (one-time, ~45 MB total):

```bash
mkdir data
curl -o data/epicure_cooc.csv https://arxiv.org/src/2605.22391v1/anc/epicure_cooc.csv
curl -o data/epicure_chem.csv https://arxiv.org/src/2605.22391v1/anc/epicure_chem.csv
curl -o data/epicure_core.csv https://arxiv.org/src/2605.22391v1/anc/epicure_core.csv
```

Or run without installing:

```bash
python epicure.py pair salmon miso
```

## Usage

```bash
# Search the 1,790-ingredient vocabulary
epicure search pepper

# What pairs well with salmon and miso? (uses molecular compatibility by default)
epicure pair salmon miso

# Surprising pairings: good molecular compatibility, rarely combined in recipes
epicure surprise chocolate

# What to add to make it more Japanese?
epicure steer japanese salmon ginger

# Steer using relative direction between two cuisines
epicure steer --from italian --to japanese salmon garlic

# Most similar ingredients (cultural substitutes by default)
epicure similar miso

# Generate a recipe (Ollama must be running)
epicure recipe beef onion tomato

# Use a specific Ollama model
epicure recipe --model llama3.2 chicken garlic
```

## Embedding variants

| Flag | Space | Best for |
|------|-------|----------|
| `--variant cooc` | Cultural co-occurrence | `similar`, `steer` (default for both) |
| `--variant chem` | Molecular compatibility | `pair` (default) |
| `--variant core` | Cross-lingual core | `steer` for broader coverage |
| `--variant ensemble` | 40% cooc + 40% chem + 20% core | `pair`, `recipe` for balanced results |

```bash
epicure pair salmon ginger --variant ensemble
epicure similar miso --variant chem
```

## Creative / surprise mode

```bash
# Surface pairings with good molecular chemistry but unusual in recipes
epicure pair salmon --creative
epicure surprise lamb
```

## Bash pipes

Commands write one ingredient per line when stdout is not a terminal, so they chain with standard Unix tools:

```bash
epicure pair salmon miso | epicure steer japanese | epicure recipe
epicure pair chicken garlic lemon | epicure recipe
epicure surprise chocolate | epicure recipe

# Works with xargs and while-read too
epicure pair beef onion | while read ing; do echo "  $ing"; done
```

Status output (loading messages, ingredient lists) always goes to stderr. Suppress with `-q`:

```bash
epicure -q pair salmon miso | epicure -q recipe
```

## Cuisines

`italian` · `mexican` · `japanese` · `indian` · `chinese` · `thai` · `french` · `mediterranean` · `korean` · `middle_eastern`

## How it works

All three embedding files (300 dimensions, 1,790 ingredients) are skip-gram vectors trained with [Metapath2Vec](https://dl.acm.org/doi/10.1145/3097983.3098036) on graphs built from 4.1M recipes, differing in what graph edges represent.

- **IDF-weighted centroid:** input ingredients are averaged with `log(N/rank)` weights so common ingredients (salt, garlic) don't dominate the query vector.
- **Hubness correction:** per-ingredient mean similarity is precomputed and subtracted from raw scores, preventing universally high-similarity "hub" ingredients from flooding results.
- **MMR diversity:** Maximal Marginal Relevance (λ=0.6) balances relevance and diversity in the top-k results.
- **Slerp:** `steer` uses spherical linear interpolation between the ingredient centroid and a discriminative cuisine pole (global corpus mean subtracted to isolate cuisine-specific direction).
- **Surprise:** z-normalised `chem_z − cooc_z` surfaces scientifically compatible but culturally unusual pairings.
- **Ensemble:** z-normalised weighted sum `0.4·cooc + 0.4·chem + 0.2·core`.

## Sources

- **Embeddings:** Radzikowski, J. & Chen, J. (2026). *Epicure: Food Ingredient Embeddings Expose the Chemistry–Recipe-Context Trade-off as a Controllable Design Axis.* arXiv:2605.22391. Embeddings used under the terms of the arXiv submission (ancillary files).
- **Flavor compounds:** Garg, N. et al. (2018). [FlavorDB: a database of flavor molecules](https://academic.oup.com/nar/article/46/D1/D1210/4559748). *Nucleic Acids Research*, 46(D1), D1210–D1216.
- **Recipe data:** Epicure aggregates from 11 public datasets including Recipe1M, Yummly, and others listed in the paper's supplement.
