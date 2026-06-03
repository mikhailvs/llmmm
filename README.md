# llmmm

Offline ingredient pairing and recipe generation, powered by food embeddings from the [Epicure paper](https://arxiv.org/abs/2605.22391).

Embeddings encode culinary relationships learned from 4.1 million recipes across 7 languages. The tool finds ingredient pairings using vector similarity, steers toward cuisine styles using spherical interpolation, and generates recipes with a local LLM via Ollama — no internet required after setup.

## Setup

**Requirements:** Python 3.10+, numpy, [Ollama](https://ollama.com) (for recipe generation)

```bash
pip install numpy
ollama pull qwen3:8b
```

Download the embeddings (one-time, ~15 MB):

```bash
mkdir data
curl -o data/epicure_cooc.csv https://arxiv.org/src/2605.22391v1/anc/epicure_cooc.csv
```

## Usage

```bash
# What pairs well with chicken, garlic, and lemon?
python epicure.py pair chicken garlic lemon

# What to add to make it more Japanese?
python epicure.py steer japanese salmon ginger

# Ingredients most similar to miso
python epicure.py similar miso

# Search the 1,790-ingredient vocabulary
python epicure.py search pepper

# Generate a recipe (Ollama must be running)
python epicure.py recipe beef onion tomato
```

## Bash pipes

Commands read from stdin and write plain ingredient names when piped, so they chain:

```bash
# Pair → recipe
echo "chicken garlic lemon" | python epicure.py pair | python epicure.py recipe

# Steer → recipe
echo "salmon miso" | python epicure.py steer japanese | python epicure.py recipe

# Full pipeline
python epicure.py pair beef onion | python epicure.py steer french | python epicure.py recipe
```

Status output (ingredient lists, loading messages) always goes to stderr so it doesn't interfere with piping.

## Cuisines

`italian` · `mexican` · `japanese` · `indian` · `chinese` · `thai` · `french` · `mediterranean` · `korean` · `middle_eastern`

## Model

Defaults to `qwen3:8b` via Ollama. Override with:

```bash
EPICURE_MODEL=llama3.2 python epicure.py recipe chicken garlic
```

Any model available in your Ollama installation works.

## How it works

The embeddings (300 dimensions, 1,790 ingredients) are skip-gram vectors trained with [Metapath2Vec](https://dl.acm.org/doi/10.1145/3097983.3098036) on a co-occurrence graph built from 4.1M recipes. `pair` and `similar` find nearest neighbors by cosine similarity. `steer` uses [SLERP](https://en.wikipedia.org/wiki/Slerp) to interpolate between your ingredient centroid and a cuisine pole (the centroid of representative cuisine markers), then returns nearest neighbors to the steered point.

## Sources

- **Embeddings:** Radzikowski, J. & Chen, J. (2026). *Epicure: Food Ingredient Embeddings Expose the Chemistry–Recipe-Context Trade-off as a Controllable Design Axis.* arXiv:2605.22391. Embeddings used under the terms of the arXiv submission (ancillary files).
- **Flavor compounds:** Garg, N. et al. (2018). [FlavorDB: a database of flavor molecules](https://academic.oup.com/nar/article/46/D1/D1210/4559748). *Nucleic Acids Research*, 46(D1), D1210–D1216.
- **Recipe data:** Epicure aggregates from 11 public datasets including Recipe1M, Yummly, and others listed in the paper's supplement.
