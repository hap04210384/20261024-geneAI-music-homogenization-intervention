# Reproduction package — melodic-pattern transmission through generative models

This repository contains the code, reference libraries, and protocols needed to
reproduce the computational experiments reported in **the manuscript** (title
and venue intentionally not stated here; both may change during review and
will be linked upon publication).

The study formalises a melodic tradition as a library of windowed melodic
patterns with a document-frequency distribution (a "cultural genotype"),
treats generative models as mediating transformations of that genotype, and
tests three propositions — cross-cultural convergence, structured collapse of
pattern diversity, and restorability of cultural fidelity by zero-training
pattern injection at inference time — on two symbolic corpora across three
families of large language models.

---

## 1. Repository layout

```
src/
  encode/     corpus normalisation and windowing (event streams, encodings a1-a3)
  mine/       threshold-free exact pattern mining driver
  validate/   engine verification against exhaustive enumeration
  gen/        LLM sampling, manipulation checks, null models, P1-P3 statistics,
              seeding interventions, figure/table generation
reference-libraries/
  reflib.{mtc,pop909}.{a1,a2,a3}.json.gz   six checksum-annotated pattern
              libraries mined from the human corpora (the "reference genotypes")
manifests/
  reference-libraries.md5   MD5 checksums of the six libraries above
  protocol-fingerprint.json vendor snapshot names, API request windows, prompt
                            templates (verbatim, with checksums), decoding parameters,
                            and per-file code checksums for the generation stages
docs/
  experiment_plan.md, paper_framework.md, paper_number_lineage.md,
  writing_conventions.md
```

## 2. Corpora — download them yourself

**No corpus audio or score files are distributed in this repository.** Both
corpora are publicly available under their own terms, which do not permit
re-hosting here. Obtain them from the official sources:

| Corpus | Contents | Official source |
|---|---|---|
| Meertens Tune Collections, release MTC-FS-INST-2.0 | 18,109 Dutch and Flemish folk songs and dance tunes (feature-sequence distribution) | https://www.liederenbank.nl/mtc/ |
| POP909 | 909 Chinese popular songs, aligned MIDI with dedicated vocal-melody track | https://github.com/music-x-lab/POP909-Dataset |

Place the downloaded material in a local `data/` directory (git-ignored) and
run the encoding scripts in `src/encode/` to produce the unified event-stream
format used by the mining stage. The **reference libraries in
`reference-libraries/` are already mined and checksum-annotated**, so every
statistic reported in the manuscript can be recomputed bit-for-bit from them
**without re-running the mining stage**; re-mining from raw corpora is only
needed to validate the libraries end-to-end.

## 3. Reproducing the experiments

1. **Environment**: Python 3.10+; see `requirements` notes inside each
   `src/` subfolder. The mining stage uses the companion anytime exact-mining
   engine (used unchanged; its internals are reported in separate companion
   work and are not re-claimed here).
2. **Model sampling (P1-P3)**: three commercial LLM families are accessed
   through their public APIs. API keys are **not** included; copy
   `.env.example` to `.env` and fill in your own keys. Snapshot names, request
   dates, prompts, and decoding parameters are recorded in the protocol
   fingerprints produced by the sampling scripts (see `src/gen/protocol_*.json`).
3. **Order of execution**: encode -> mine/validate -> gate & sample ->
   P1 -> P2 -> P3 -> aggregate. Each stage writes JSON results and a manifest;
   the aggregators in `src/gen/` turn them into the manuscript's tables and
   figures. Stage entry points are the `stage_*.py` / `sample*.py` scripts in
   `src/gen/`.
4. **Determinism**: the pipeline is deterministic — identical input bytes
   produce identical libraries; verify any downloaded reference library
   against `manifests/reference-libraries.md5`.

## 4. Licensing and anonymity

- Reference libraries are released for reproduction and review.
- During double-anonymous review this repository is mirrored through an
  anonymous access service; the canonical repository (with author identity)
  will be linked at publication.
- The corpora remain the property of their respective distributors; cite them
  as instructed by their own terms when using this package.
