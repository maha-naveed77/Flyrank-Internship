# Capstone Report

* Author: Maham Naveed
* Lane: Refresh / Content Opportunity Scoring
* Repo: https://github.com/maha-naveed77/Flyrank-Internship
* Date: September 2026

## 0. Abstract
Which pages should a content reviewer with limited time check first for a possible refresh? This work uses FlyRank's search performance warehouse, aggregating April 2026 search-console signals to predict a >20% impression drop the following month (May 2026). A Random Forest model, validated under an honest client-grouped split, reaches roughly 0.80 Precision@50, compared to roughly 0.28 for a hand-written baseline rule on the same split against a base rate of ~50% decline in the underlying population. The output feeds a three-tier, reason-coded action queue intended as decision-support for a content reviewer, not an autonomous or causal system.

## 1. Problem framing
**Decision:** which pages get prioritized for manual refresh review out of a large backlog, given limited reviewer capacity (e.g. ~50 pages/cycle).
**Unit of analysis:** one content page, belonging to one client, aggregated over one calendar month.
**Output:** a ranked decline-risk score per page, converted into a three-tier action label (`review_for_refresh`, `monitor`, `no_action`) with a reason code.
**Action a human takes:** a reviewer works down the ranked queue and manually inspects flagged pages before deciding whether to refresh.
**Cost of a wrong call:** two-sided flagging a stable page wastes a limited review slot; missing a genuinely declining, high-traffic page lets it keep losing visibility silently until the next cycle.
**Why ML helps:** roughly half of all pages show some decline signal in a given month far more than any reviewer can check by hand — and a fixed hand-written rule applies the same weights to every page regardless of how signals actually interact; the ~3x Precision@50 gap over the baseline shows those interactions matter.

## 2. Data safety
**Data used:** FlyRank ML Internship warehouse (Hugging Face: `FlyRank/internship-warehouse`), table `fact_content_daily_performance`, months `2026-04` (features) and `2026-05` (label window only).
**Deliberately excluded:** the sealed final month (June 2026, `_sample` table) excluded from all development since it's the natural outcome window of any past→future label and would leak the answer if used for anything beyond mechanics testing. Any FlyRank product-decision fields (health/priority/optimization-flag style columns) were excluded as features, since they represent existing business rules, not raw observed signal, and would let the model copy a decision rather than learn anything new.
**Leakage risks considered:** confirmed all six model features (`impressions_apr`, `clicks_apr`, `avg_position_apr`, `ctr_apr`, `sessions_ai_apr`, `scroll_events_apr`) come only from the April window, strictly before the May label window no feature name references May or any future period. `client_hash_id` and `content_hash_id` are used only for grouping and joins, never as model features.
**Client-identifying data:** none appears anywhere in `work/` all IDs are pre-hashed by the warehouse itself, and no raw client names, domains, or URLs were introduced at any stage.

## 3. Baseline
A transparent, hand-written rule: `baseline_score = (expected_ctr_for_position_tier − actual_ctr) × impressions`, where `expected_ctr_for_position_tier` is the training-set average CTR for pages at the same position tier (top_3 / page_1 / striking / page_3_5 / deep). This is a fair comparison because it's evaluated on the identical client-grouped test split and the identical Precision@50 metric as the model. On that split, the baseline scores approximately **0.28** Precision@50.

## 4. Model / analysis
**Method:** Random Forest classifier. Chosen because the Week-4 signal audit found staleness vs. CTR to be a MIXED, non-monotonic relationship suggesting decline risk depends on combinations of signals rather than a single linear rule, which a tree ensemble can capture and a hand-written weighted sum cannot.
**Features:** `impressions_apr`, `clicks_apr`, `avg_position_apr`, `ctr_apr`, `sessions_ai_apr`, `scroll_events_apr` all aggregated from April only.
**Left out on purpose:** any FlyRank product/decision field, and raw client/content IDs as features (grouping keys only).
**Target/proxy, in one sentence:** `is_declining = 1` if a page's May 2026 impressions fell more than 20% below its April 2026 impressions a proxy for decline, not a verified causal outcome.

## 5. Evaluation
**Split:** grouped by `client_hash_id` (80% of clients train, 20% test; ~41 vs. ~10 clients depending on run) — no client's pages appear in both sets, since pages from the same client share systemic quirks that a random row-level split would let the model memorize instead of generalizing from.
**Base rate:** `is_declining` is close to balanced in this population (~50%), so a high Precision@50 is not simply reflecting a skewed majority class.
**Metrics — model vs. baseline, same split:**

| Method | Precision@50 |
|---|---|
| Baseline rule | ~0.28 |
| Random Forest (grouped split) | ~0.80 |

A naive random (non-grouped) split was also tested for comparison and produced an inflated Precision@50 (as high as 0.90) — direct evidence that the grouped split is the fairer, honest number, since the naive split let the model partly memorize client-specific patterns.
**Error analysis:** among the model's top-ranked pages, most were true positives; the false positives that did occur were consistently very small pages (low monthly impressions, near-zero CTR) at weak average positions — plausibly already-marginal pages with little room left to decline, rather than pages showing a genuine emerging decline pattern.

## 6. Interpretation
Permutation importance ranks `ctr_apr` and `avg_position_apr` as the strongest observed drivers of the model's decline-risk score, with `impressions_apr` and `clicks_apr` close behind. `sessions_ai_apr` and `scroll_events_apr` contribute almost nothing (near-zero or slightly negative importance) — consistent with their very low (~4%) availability across this warehouse slice, not necessarily because engagement is truly irrelevant to decline. **Negative result worth stating plainly:** the Week-4 signal audit found staleness (days since last update) alone to be a MIXED, non-monotonic signal — the `90-180 day` staleness bucket actually had the *highest* CTR of any bucket in that audit, not the lowest — so "how old is this content" is not, on its own, a reliable predictor in this dataset, contrary to a common SEO assumption.

## 7. Recommendation
The model's score converts into a three-tier action queue: `review_for_refresh` (score ≥ 0.6), `monitor` (0.4–0.6), `no_action` (below 0.4), each tagged with a reason code (`HIGH_VISIBILITY_LOW_CTR`, `POOR_POSITION_HIGH_IMPRESSIONS`, `GENERAL_DECLINE_RISK`) so a reviewer understands *why* a page was flagged. A FlyRank editor would use this tomorrow by pulling the `review_for_refresh` tier at the start of a review cycle and working down it in ranked order rather than reviewing pages in an arbitrary or purely chronological order.
**Confidence and limits, stated explicitly:** this is decision-support only, validated on a single month pair and one client split — not a causal or production-certified system. It should never auto-publish changes, auto-deprioritize, or be presented to a client as a performance guarantee. A human must still check whether a flagged page's low CTR reflects genuine decline versus a naturally low-CTR content type.

## 8. Reproducibility
**Environment:** Google Colab (Python 3, standard scientific stack — `pandas`, `numpy`, `scikit-learn`, `duckdb`, `matplotlib`).
**To re-run from a fresh clone:**
```bash
git clone https://github.com/maha-naveed77/Flyrank-Internship
cd Flyrank-Internship
pip install -r requirements.txt
# open work/notebooks/capstone.ipynb in Colab, set HF_TOKEN as a Colab Secret, Runtime → Run all
```
**Random seed:** `random_state=42` used throughout (train/test client sampling and `RandomForestClassifier`).
**Sealed evaluation:** the final June 2026 month (`fact_content_daily_performance_sample.parquet`) was never used to build labels, tune features, or select the model — only touched, if at all, for query-mechanics testing per the internship's own data-use guidance. The cell that builds the April/May training frame and the `work/outputs/model_metrics.json` file it produces are both committed, so the client-grouped Precision@50 numbers reported here can be checked directly against a fresh run rather than taken on faith.
**Known variability:** repeated runs of this same pipeline showed Precision@50 varying in a small range (baseline ~0.24–0.28, model ~0.80–0.82) due to `RandomForestClassifier`'s internal randomness and the specific random client sample drawn each run for the test split — the ~3x model-over-baseline gap was stable across every run; the exact third-decimal figures were not.

## 9. Acknowledgments & data credit
Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai)
