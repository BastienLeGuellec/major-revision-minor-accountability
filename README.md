# Major Revision, Minor Accountability: LLMs as Reviewers of Published Articles in Top-Tier Journals

Code, corpus list and raw outputs for the study by Bastien Le Guellec, Su-Hwan Kim, Raphael Bentegeac and Aghiles Hamroun.

Four large language models were asked to peer-review 200 articles already published in 2026, under a Creative Commons Attribution (CC BY) license, in 20 general-science, life-science and clinical-medicine journals (ten per journal). Each model was told it was a reviewer for the article's real journal and was not told the article had been published. The study covers three experiments plus a revision-framing comparison:

| Manuscript | What it measures | Script | Output |
|---|---|---|---|
| Experiment 1 (§3.1) | Decision on each unmodified manuscript, 4 models × 200 | `experiments/exp1_accept_rate/run.py` | `results/exp1_decisions.csv` |
| Revision framing (§3.2) | Same, with the prompt saying the manuscript is a revision | `experiments/exp1_accept_rate/run_second_round.py` | `results/exp1_secondround.csv` |
| Repeat review (reported by `manuscript_stats.py`) | Identical second pass, for intra-model agreement | `experiments/exp1_accept_rate/run_intra_rater.py` | `results/exp1_intra_rater_retest.csv` |
| Experiment 2 (§3.3) | Detection of one planted flaw per manuscript | `experiments/exp2_error_injection/run.py` | `results/exp2_injections.jsonl` |
| Experiment 3 (§3.4) | Blind judging of the four Experiment-1 reviews of each manuscript | `experiments/exp5_review_quality/run.py` | `results/exp5_review_quality.jsonl` (Grok 4.6), `results/exp5_review_quality_minimax.jsonl` (MiniMax M3) |

The review-quality experiment lives in `exp5_review_quality/` for historical reasons; it is Experiment 3 in the manuscript.

## Reproduce every number and figure (no API key needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 figures/manuscript_stats.py     # every statistic quoted in the manuscript, by section
python3 figures/make_figures.py         # regenerates the data figures into figures/
```

Tested with Python 3.12. Figures use the Liberation Sans font (`fonts-liberation` on Debian/Ubuntu); matplotlib falls back to its default font if it is missing.

| Manuscript figure | File | Made by |
|---|---|---|
| Figure 1 (study design) | `figures/fig_study_design_v2.png` | drawn by hand, not generated |
| Figure 2 | `figures/fig1_status_by_model.png` | `make_figures.py` |
| Figure 3 | `figures/fig2_status_all_papers.png` | `make_figures.py` |
| Figure 4 | `figures/fig9_second_round.png` | `make_figures.py` |
| Figure 5 | `figures/fig4_exp2_detection.png` | `make_figures.py` |
| Figure 6 | `figures/fig5_best_review.png` | `make_figures.py` |
| Figure 7 | `figures/fig7_concern_overlap.png` | `make_figures.py` |

`make_figures.py` also writes `fig8_ngram_fingerprint.png`, a phrase-frequency comparison that is not part of the manuscript.

## Re-run the experiments

The article texts are not included (see [Corpus](#corpus)), so rebuild them first; this needs `SPRINGER_NATURE_API_KEY` (free from the Springer Nature API portal). All model calls go through [OpenRouter](https://openrouter.ai) and need `OPENROUTER_API_KEY`. Copy `.env.example` to `.env` and set both. Every runner is idempotent: it skips any (manuscript, model) pair already present in its output file, so delete or move the corresponding `results/` file to start over.

```bash
python3 experiments/common/rebuild_corpus.py                  # papers/*.txt and papers/_exp2_injected/*.txt
python3 experiments/exp1_accept_rate/run.py                   # Experiment 1
python3 experiments/exp1_accept_rate/run_second_round.py      # revision framing
python3 experiments/exp1_accept_rate/run_intra_rater.py --dois-file results/intra_rater_full_sample.json
python3 experiments/exp2_error_injection/run.py               # injection, review and detection judging
python3 experiments/exp2_error_injection/rerun_unparsed.py    # retry reviews with no parseable decision
python3 experiments/exp5_review_quality/run.py --judge-model x-ai/grok-4.6
python3 experiments/exp5_review_quality/run.py --judge-model minimax/minimax-m3 \
        --out results/exp5_review_quality_minimax.jsonl
```

Models (OpenRouter identifiers):

| Role | Model | Settings |
|---|---|---|
| Reviewer panel | `z-ai/glm-5.3-flash`, `openai/gpt-5.6-sol`, `google/gemini-3.7-flash`, `deepseek/deepseek-v4-flash-0731` | temperature 0.7, 24,000 max tokens, reasoning capped at a third of that |
| Flaw injector (Exp. 2) | `x-ai/grok-4.6` | temperature 0.9, reasoning effort low |
| Detection judge (Exp. 2) | `anthropic/claude-opus-5` | temperature 0, reasoning effort low |
| Quality judges (Exp. 3) | `x-ai/grok-4.6`, `minimax/minimax-m3` | temperature 0, reasoning effort low, 4,000 max tokens (16,000 for six MiniMax re-runs) |

The four reviewer models are the default for every runner (`llm_clients.PANEL_MODELS`). Re-running will not reproduce the outputs exactly: reviews are sampled at temperature 0.7, and a provider can change the model served behind an identifier. At October 2026 prices one 800-review pass over the corpus costs about US$11, most of it GPT-5.6 Sol.

DeepSeek v4 Flash occasionally returns an empty or truncated answer on the longest manuscripts. Check row counts after every run and re-run the script to fill gaps.

## Corpus

The article texts are not redistributed here; only their identifiers are. `papers/corpus.csv` lists the 200 articles with DOI, journal, title, license, copyright holder, and the SHA-256 of the exact text each model reviewed (`text_sha256`) and of its flaw-injected version (`injected_sha256`). `papers/_journal_manifest.json` maps each DOI to its journal and `results/final_corpus_dois.json` lists the 200 DOIs.

Inclusion criteria: published in 2026; a CC BY license; body of at least 2,000 words; a research or review article (no correction, erratum, retraction or short-format letter). Full text was retrieved as JATS XML from the Springer Nature Open Access API for Nature, Nature Medicine, Nature Communications, Nature Genetics and Scientific Reports, and from Europe PMC for the other 15 journals (`experiments/common/get_fulltext.py`, `get_europepmc_papers.py`, `fetch_ccby_topup.py`, `fetch_replacements.py`, filtered by `article_filter.py`). The title, abstract, body text, figure and table captions, and in-XML table content were concatenated into plain text; images and supplementary files were not included. `experiments/common/check_licenses.py` re-reads each article's license from its source XML into `results/license_check.jsonl`.

`experiments/common/rebuild_corpus.py` re-fetches every article with the same extractors, writes `papers/<doi with / replaced by _>.txt`, rebuilds each flaw-injected text in `papers/_exp2_injected/` by applying the `flaw_location` replacement recorded in `results/exp2_injections.jsonl`, and checks both against the SHA-256 in `corpus.csv`. Tested on 23 October 2026, 147 of the 200 texts rebuilt byte for byte. Most of the others differ in formatting because they were extracted by earlier versions of the code (abstract labels, inline citation numbers, table layout, supplement captions), and one article is no longer served by Europe PMC. The script lists every text that differs. Some DOIs contain several slashes (for example `10.1093/eurheartj/...`), so recover a DOI from a file name with `common.journal_manifest.doi_for_stem()`, not by string replacement.

The rebuilt files in `papers/_exp2_injected/` are **deliberately altered**: each contains a serious methodological or statistical flaw and does not describe the study as it was conducted or published. Do not read, cite or reuse them as the original articles.

## Output formats

- `exp1_*.csv`: `doi, model, decision, parse_method, word_count, raw_review`. The decision is `accept`, `minor_revision`, `major_revision` or `reject`; `parse_method` says whether it came from the explicit `DECISION:` line or a keyword fallback.
- `exp2_injections.jsonl`: one record per (manuscript, reviewer). Fields: `category` (statistical or methodological), `central_conclusion`, `flaw_summary`, `flaw_location` (`original -> replacement`), `raw_review`, and `detected` (`true`, `false` or `"partial"`). Only `true` counts as detection in the manuscript.
- `exp5_review_quality*.jsonl`: one record per manuscript. Fields: `letter_to_model` (the per-manuscript random A–D assignment), `concern_clusters` (`summary`, `raised_by`, `category` relevant/nitpicking), `constructiveness` (1–5 per letter), `best_review_letter`, `best_review_model` and `best_review_reason`.

## License

- Code: MIT (`LICENSE`).
- Article texts: not included. Each article remains under its publisher's CC BY license (version and attribution per article in `papers/corpus.csv`; see `DATA_LICENSE.md`).
- Model outputs and judgments in `results/`: CC BY 4.0.

## Citation

See `CITATION.cff`. The manuscript reference will be added on publication.
