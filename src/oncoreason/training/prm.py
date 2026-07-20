"""Process Reward Model (PRM) — the trained verifier.

Input (case, step, evidence) -> P(step sound). Trained on the guideline-verified step labels
from Phase 4. Two backends behind one interface:

  - ``tfidf`` (default): TF-IDF + logistic regression, scikit-learn. Trains and scores on CPU
    in seconds, so the offline/CI path is real rather than mocked. This is a legitimate
    baseline verifier, and it is what the reported numbers come from until the transformer
    path is run.
  - ``modernbert``: ModernBERT-base (149M, 8k ctx) fine-tuned as a step-soundness classifier.
    Needs `transformers` + a GPU; runs on Kaggle. Raises a clear enable-message otherwise.

Non-negotiables implemented here (Phase 5.1):
  - **CASE-LEVEL splits** — a case never appears in both train and test (step-level splitting
    leaks, because steps from one case share evidence and phrasing).
  - **class imbalance** handled via balanced class weights (sound steps dominate).
  - **temperature scaling** so the score is calibrated, which is what makes it usable as a
    confidence signal for abstention downstream.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field


@dataclass
class PRMConfig:
    backend: str = "tfidf"                       # "tfidf" (CPU) | "modernbert" (Kaggle GPU)
    base_model: str = "answerdotai/ModernBERT-base"
    max_length: int = 4096
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 8
    seed: int = 17
    temperature_scale: bool = True
    test_frac: float = 0.3


@dataclass
class PRMReport:
    """What Phase 5.1 must report: a confusion matrix on a held-out CASE-level split."""
    n_train: int
    n_test: int
    n_train_cases: int
    n_test_cases: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    temperature: float
    positive_rate: float
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return dict(self.__dict__)


def split_by_case(examples: list[dict], test_frac: float = 0.3, seed: int = 17):
    """Split step-examples so that **no case appears on both sides** (leakage guard).

    Splitting at the step level would put steps from the same case in train and test; they
    share evidence, phrasing and topic, so the verifier would be scored on near-duplicates.
    """
    cases = sorted({e["case_id"] for e in examples})
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_test = max(1, int(len(cases) * test_frac)) if len(cases) > 1 else 0
    test_cases = set(cases[:n_test])
    train = [e for e in examples if e["case_id"] not in test_cases]
    test = [e for e in examples if e["case_id"] in test_cases]
    return train, test, test_cases


def _featurize(e: dict) -> str:
    """The PRM sees the step, the ids it cites, and — when supplied — what those records say.

    Ids alone are opaque tokens: `civic:EID2994` carries no information about whether the
    record supports the claim, so a model given only ids can learn citation *presence* but
    never citation *correctness*. That ceiling applies to any backend, which is why the
    ModernBERT path has little to gain from an 8k context until the text is actually there.

    `evidence_text` is optional, so callers holding only ids keep their previous behaviour.
    """
    ev = " ".join(e.get("evidence_ids") or []) or "NO_EVIDENCE"
    base = f"{e.get('step_text', '')} [EVIDENCE] {ev}"
    txt = " ".join(e.get("evidence_text") or [])
    return f"{base} {txt}" if txt else base


def _fit_temperature(logits: list[float], labels: list[int]) -> float:
    """Temperature scaling: find T>0 minimising NLL of sigmoid(logit / T).

    Calibration is the point: an uncalibrated verifier score cannot drive abstention.
    A simple 1-D search is ample at these sizes.
    """
    best_t, best_nll = 1.0, float("inf")
    for i in range(1, 501):
        t = i / 100.0
        nll = 0.0
        for z, y in zip(logits, labels):
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z / t))))
            p = min(max(p, 1e-9), 1 - 1e-9)
            nll -= y * math.log(p) + (1 - y) * math.log(1 - p)
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


class PRM:
    """A trained step-soundness verifier. `predict_proba` returns calibrated P(sound)."""

    def __init__(self, cfg: PRMConfig | None = None):
        self.cfg = cfg or PRMConfig()
        self._vec = None          # tfidf backend
        self._clf = None
        self._tok = None          # modernbert backend
        self._model = None
        self.temperature = 1.0

    # -- training ------------------------------------------------------------------
    def fit(self, examples: list[dict]) -> "PRM":
        if self.cfg.backend == "modernbert":
            self._fit_modernbert(examples)
            return self
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        X = [_featurize(e) for e in examples]
        y = [1 if e["label_sound"] else 0 for e in examples]
        self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        Xv = self._vec.fit_transform(X)
        # balanced weights: sound steps dominate the label distribution
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=self.cfg.seed)
        self._clf.fit(Xv, y)
        return self

    def _fit_modernbert(self, examples):  # pragma: no cover - GPU path
        """Fine-tune ModernBERT-base as a step-soundness classifier.

        A plain loop rather than `transformers.Trainer`: the training signal here is one
        binary head over a few hundred short sequences, so Trainer's machinery buys nothing
        and its API moves between releases. Class imbalance is handled with a positive
        weight in the loss, mirroring the `class_weight="balanced"` of the TF-IDF path so the
        two backends stay comparable.
        """
        try:
            import torch
            from torch.utils.data import DataLoader
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise NotImplementedError(
                "backend='modernbert' needs `transformers` and a GPU (run on Kaggle). "
                f"Use backend='tfidf' for the CPU path. Base model: {self.cfg.base_model}."
            ) from exc

        torch.manual_seed(self.cfg.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tok = AutoTokenizer.from_pretrained(self.cfg.base_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.base_model, num_labels=1).to(device)

        texts = [_featurize(e) for e in examples]
        labels = [1.0 if e["label_sound"] else 0.0 for e in examples]
        n_pos = sum(labels) or 1.0
        pos_weight = torch.tensor([(len(labels) - n_pos) / n_pos], device=device)

        idx = list(range(len(texts)))
        loader = DataLoader(idx, batch_size=self.cfg.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.cfg.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self._model.train()
        for epoch in range(self.cfg.epochs):
            total = 0.0
            for batch in loader:
                b = [int(i) for i in batch]
                enc = self._tok([texts[i] for i in b], truncation=True,
                                max_length=self.cfg.max_length, padding=True,
                                return_tensors="pt").to(device)
                y = torch.tensor([labels[i] for i in b], device=device).unsqueeze(1)
                loss = loss_fn(self._model(**enc).logits, y)
                loss.backward()
                opt.step()
                opt.zero_grad()
                total += float(loss)
            print(f"  epoch {epoch + 1}/{self.cfg.epochs}  mean loss {total / len(loader):.4f}")
        self._model.eval()
        return self

    def _scores_modernbert(self, examples: list[dict]) -> list[float]:  # pragma: no cover
        import torch

        device = next(self._model.parameters()).device
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(examples), self.cfg.batch_size):
                chunk = examples[i:i + self.cfg.batch_size]
                enc = self._tok([_featurize(e) for e in chunk], truncation=True,
                                max_length=self.cfg.max_length, padding=True,
                                return_tensors="pt").to(device)
                out += [float(z) for z in self._model(**enc).logits.squeeze(-1)]
        return out

    # -- scoring -------------------------------------------------------------------
    def decision_scores(self, examples: list[dict]) -> list[float]:
        if self.cfg.backend == "modernbert":
            return self._scores_modernbert(examples)
        Xv = self._vec.transform([_featurize(e) for e in examples])
        return [float(z) for z in self._clf.decision_function(Xv)]

    def predict_proba(self, examples: list[dict]) -> list[float]:
        """Calibrated P(step sound)."""
        out = []
        for z in self.decision_scores(examples):
            zt = max(-30.0, min(30.0, z / (self.temperature or 1.0)))
            out.append(1.0 / (1.0 + math.exp(-zt)))
        return out

    def score_step(self, step_text: str, evidence_ids: list[str] | None = None) -> float:
        return self.predict_proba([{"step_text": step_text,
                                    "evidence_ids": evidence_ids or []}])[0]

    # -- persistence ---------------------------------------------------------------
    def save(self, out_dir: str) -> str:
        import pickle

        os.makedirs(out_dir, exist_ok=True)
        if self.cfg.backend == "modernbert":  # pragma: no cover - GPU path
            self._model.save_pretrained(out_dir)
            self._tok.save_pretrained(out_dir)
            with open(os.path.join(out_dir, "prm_meta.json"), "w") as f:
                json.dump({"temperature": self.temperature,
                           "backend": self.cfg.backend,
                           "base_model": self.cfg.base_model}, f, indent=1)
            return out_dir
        with open(os.path.join(out_dir, "prm.pkl"), "wb") as f:
            pickle.dump({"vec": self._vec, "clf": self._clf,
                         "temperature": self.temperature, "cfg": self.cfg}, f)
        return out_dir

    @classmethod
    def load(cls, model_dir: str) -> "PRM":
        import pickle

        meta = os.path.join(model_dir, "prm_meta.json")
        if os.path.exists(meta):  # pragma: no cover - GPU path
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            blob = json.load(open(meta))
            m = cls(PRMConfig(backend="modernbert", base_model=blob["base_model"]))
            m._tok = AutoTokenizer.from_pretrained(model_dir)
            m._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            m._model.eval()
            m.temperature = blob["temperature"]
            return m

        with open(os.path.join(model_dir, "prm.pkl"), "rb") as f:
            blob = pickle.load(f)
        m = cls(blob["cfg"])
        m._vec, m._clf, m.temperature = blob["vec"], blob["clf"], blob["temperature"]
        return m


def train_prm(examples: list[dict], cfg: PRMConfig | None = None,
              out_dir: str | None = None) -> tuple[PRM, PRMReport]:
    """Train the PRM on Phase-4 step labels and report held-out performance.

    Returns (model, report). The report carries the confusion matrix on a **case-level**
    held-out split, which is the number Phase 5.1 is required to publish.
    """
    cfg = cfg or PRMConfig()
    notes: list[str] = []
    labels = {bool(e["label_sound"]) for e in examples}
    if len(labels) < 2:
        raise ValueError(
            "PRM training needs both sound and unsound steps, but the labels are degenerate "
            f"({labels}). A deterministic scaffold cannot produce ungrounded steps; mine "
            "counterfactual negatives first (supervision.mine_negatives) or use policy samples."
        )
    train, test, test_cases = split_by_case(examples, cfg.test_frac, cfg.seed)
    if not test:
        test = train
        notes.append("too few cases for a held-out split; test == train")

    model = PRM(cfg).fit(train)

    # temperature scaling on the held-out split (calibration for downstream abstention)
    if cfg.temperature_scale:
        z = model.decision_scores(test)
        y = [1 if e["label_sound"] else 0 for e in test]
        if len(set(y)) > 1:
            model.temperature = _fit_temperature(z, y)
        else:
            notes.append("single-class held-out split; temperature left at 1.0")

    probs = model.predict_proba(test)
    y_true = [1 if e["label_sound"] else 0 for e in test]
    tp = sum(1 for p, t in zip(probs, y_true) if p >= 0.5 and t == 1)
    fp = sum(1 for p, t in zip(probs, y_true) if p >= 0.5 and t == 0)
    tn = sum(1 for p, t in zip(probs, y_true) if p < 0.5 and t == 0)
    fn = sum(1 for p, t in zip(probs, y_true) if p < 0.5 and t == 1)

    report = PRMReport(
        n_train=len(train), n_test=len(test),
        n_train_cases=len({e["case_id"] for e in train}),
        n_test_cases=len(test_cases) or len({e["case_id"] for e in test}),
        tp=tp, fp=fp, tn=tn, fn=fn,
        accuracy=(tp + tn) / len(test) if test else 0.0,
        temperature=model.temperature,
        positive_rate=sum(y_true) / len(y_true) if y_true else 0.0,
        notes=notes,
    )
    if out_dir:
        model.save(out_dir)
        with open(os.path.join(out_dir, "prm_report.json"), "w") as f:
            json.dump(report.summary(), f, indent=1)
    return model, report
