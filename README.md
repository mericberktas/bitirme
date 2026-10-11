# turkish_distill

Multi-teacher distillation for Turkish text embedding models (graduation project, Yildiz Technical University).

The student is `ytu-ce-cosmos/modernbert-tr-base-1k`. Several teacher embedding models (e.g. Qwen3-Embedding-8B, BGE-M3, multilingual-e5-large-instruct) are encoded once into an on-disk cache; the student is then trained to reproduce them (offline distillation).

## Repository layout

```
src/turkish_distill/   Python package (config, losses, models, teachers cache, training, eval)
configs/               YAML experiment definitions
notebooks/             Thin Colab notebook(s) that call the package
tests/                 CPU-only unit and smoke tests (no GPU, no model downloads)
rapor/                 LaTeX project report
arastirma/             Literature / teacher / dataset / benchmark research notes
CLAUDE.md              Project context and technical decisions
```

## Pipeline

1. `prepare_corpus`: stream a HF dataset into `corpus.jsonl` (once).
2. `encode_teachers`: encode the corpus with every teacher into float16 shards (resumable, keyed by a corpus fingerprint).
3. `train`: train the student against the cached teacher vectors (resumable; TensorBoard + `metrics.jsonl`).

```bash
python -m turkish_distill.prepare_corpus --dataset <hf-id> --text-column text --max-samples 1000000 --out corpus.jsonl
python -m turkish_distill.encode_teachers --config configs/multi_teacher_a.yaml
python -m turkish_distill.train --config configs/multi_teacher_a.yaml
```

On Colab use `notebooks/colab_pipeline.ipynb`: it mounts Drive, clones this repo, installs the package and runs the three steps. Everything it produces (cache, checkpoints, logs) lives on Google Drive.

## Losses

Per teacher `k`, with the shared student embedding `z` and a linear head `h_k` mapping it to teacher `k`'s dimension:

- cosine: `1 - cos(h_k(z), t_k)`
- similarity: MSE between in-batch Gram matrices of `z` and `t_k` (independent of the teacher dimension)
- relative similarity: hinge on the ordering of sampled pairs of pairs (Jasper & Stella, arXiv 2412.19048)
- optional spread-out regulariser on `z`

The total is the weighted sum over teachers divided by the sum of weights, so single- and multi-teacher runs have the same gradient scale.

## Run directory

```
<output_dir>/<experiment>/<run_id>/
  config.yaml  env.json (git SHA, packages, GPU)  metrics.jsonl  tb/
  checkpoints/step_XXXXXXXX.pt   (atomic writes, last N kept)
  final/                         (student backbone in HF format + teacher heads)
```

To resume after a disconnect, run the train step again with the same `--run-id`.

## Tests

```bash
pip install -e . pytest
pytest
```

The tests use a tiny backbone and fake teachers. They verify the code paths (losses, cache, checkpoint/resume, logging), not model quality.

## Reproducibility notes

- Teacher and student models are pinned by Hugging Face commit hash (`revision` in the configs). The teacher revision is stored in each cache's `meta.json` and checked when the cache is opened.
- The HF model cache is kept on the Colab local disk (`HF_HOME=/content/hf_cache`), not on Drive; only the teacher vectors, corpus, checkpoints and logs go to Drive.

## Known limitations

- Not yet run against the real models on a GPU. Teacher dimensions, prefixes and library versions in `configs/` come from model cards and must be verified on the first Colab run.
- No benchmark evaluation yet (TR-MTEB wrapper is not implemented); only the optional fast STS monitor.
- Corpus preparation does not decontaminate against benchmark test sets.
- No gradient accumulation: the similarity losses are defined on the full in-batch matrix.
