# References

The published work this project builds on. The implementation is a small-scale synthesis of
these papers' methods; it claims no original methodology. Per-paper attribution — which idea
came from where, and where this implementation departs from it — is in
[`../summary/FINDINGS.md`](../summary/FINDINGS.md).

PDFs are not redistributed here; they are third-party copyrighted material.

## Process supervision and post-training

- Yun J, Sohn J, Park J, Kim H, et al.; Moor M. *Med-PRM: Medical Reasoning Models with
  Stepwise, Guideline-verified Process Rewards.* EMNLP 2025.
  — The guideline-supplies-the-label mechanism used for step supervision (Phases 4–5).
- *Process Reward Agents for Steering Knowledge-Intensive Reasoning.* ICML 2026.
  — Process rewards applied to an agent rather than a single-pass model; selective retrieval.

## Agents, tool use and clinical evaluation

- Schmidgall S, Ziaei R, Harris C, Kim JW, Reis EP, Jopling J, Moor M. *AgentClinic: a
  multimodal benchmark for tool-using clinical AI agents.* npj Digital Medicine, 2026.
  — Evaluation template for tool-using clinical reasoning; informs the Phase 6 metric set.
- Zakka C, Shad R, Chaurasia A, Dalal AR, Kim JL, Moor M, et al. *Almanac:
  Retrieval-Augmented Language Models for Clinical Medicine.* NEJM AI, 2024.
  — Retrieval grounding and cited output; the basis of the citation-grounding metric.
- *RadAgent: a tool-using AI agent for stepwise interpretation of chest computed
  tomography.* 2026.
  — Stepwise agent interpretation with per-step auditability.
- *Agentic Systems in Radiology: Design, Applications, Evaluation, Challenges.* Review, 2025.
  — Design and evaluation considerations for clinical agent systems.

## Multimodal medical models

- Moor M, Banerjee O, Shakeri Hossein Abad Z, Krumholz HM, Leskovec J, Topol EJ,
  Rajpurkar P. *Foundation models for generalist medical artificial intelligence.*
  Nature, 2023;616(7956):259–265.
  — The generalist-medical-AI framing this project targets.
- Moor M, Huang Q, Wu S, Yasunaga M, Dalmia Y, Leskovec J, Zakka C, Reis EP, Rajpurkar P.
  *Med-Flamingo: a Multimodal Medical Few-shot Learner.* Machine Learning for Health
  (ML4H), PMLR 225:353–367, 2023.
  — Frozen-LLM-plus-encoder fusion; the architecture the roadmap's molecular modality follows.
- *Multimodal generative AI for medical image interpretation.* Nature, 2025.
  — Current state of multimodal medical interpretation.
- *SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning.*
  NeurIPS 2025 Datasets & Benchmarks.
  — Expert-curated multimodal in-context evaluation.
- *MARBLE: A Hard Benchmark for Multimodal Spatial Reasoning and Planning.* 2025.
  — Multi-step multimodal reasoning under planning constraints.

## Oncology decision-making and knowledge infrastructure

- *MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology.*
  NeurIPS 2025 Datasets & Benchmarks.
  — The non-circular gold standard this project's evaluation should adopt; see
  [`../docs/MTBBENCH_INTEGRATION.md`](../docs/MTBBENCH_INTEGRATION.md).
- *MIRIAD: Augmenting LLMs with millions of medical query-response pairs.* 2025.
  — Retrieval and data infrastructure at scale.

## Data sources

- cBioPortal for Cancer Genomics — molecular profiles. <https://www.cbioportal.org>
- CIViC (Clinical Interpretation of Variants in Cancer) — curated clinical evidence.
  <https://civicdb.org>
- ClinVar (NCBI) — variant interpretation records.
  <https://www.ncbi.nlm.nih.gov/clinvar/>
- ClinGen Allele Registry — canonical allele identifiers (CAIDs) used as join keys.
  <https://reg.clinicalgenome.org>
