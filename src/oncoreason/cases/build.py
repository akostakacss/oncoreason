"""Case construction (Phase 2, steps 4-6): cBioPortal profiles -> annotated, split Cases.

Pipeline, in order:
  4. sample real per-tumor profiles from allow-listed studies, QC-filtered + patient-deduped
  5. annotate each alteration with CIViC (somatic) + ClinVar (germline), joined on canonical ids
  6. assemble Case objects and split at the patient level (a case == one patient here)

Design decisions are documented in docs/DATA_HANDLING.md — this module is their implementation.
Gold labels are NOT set here (that is Phase 3); cases carry attached *evidence*, not verdicts —
turning evidence into an ESCAT-tiered recommendation is interpretation, not a lookup.
"""
from __future__ import annotations

import random
from collections import defaultdict

from ..datasources import EvidenceKind, EvidenceQuery
from .qc import run_qc
from .schema import Alteration, Case, ClinicalContext, GoldRecommendation, Split

# study id -> tumor type for the case context (the one clinical field I trust: histology)
_STUDY_TUMOR_TYPE = {
    "luad_tcga_pan_can_atlas_2018": "lung adenocarcinoma",
    "lusc_tcga_pan_can_atlas_2018": "lung squamous cell carcinoma",
}

# QC checks whose records I DROP (they corrupt aggregation) vs. those I keep-but-flag.
_DROP_CHECKS = {"duplicate_row", "vaf_zero_alt"}


# ---- step 4: profile sampling -------------------------------------------------------------
def build_profiles(src, studies, gene_entrez, *, seed=17, per_study=30, min_alterations=1):
    """Group QC-passed driver mutations into per-tumor profiles.

    - drops records flagged by a `_DROP_CHECKS` QC rule (duplicates, zero-VAF)
    - keeps type_vs_protein-flagged records but records the flag on the profile (the variant is
      real; only its consequence *label* is suspect)
    - patient-level dedup within AND across studies (one sample per patient) — kills the
      TCGA-reprocessing leakage class before it can reach a split
    """
    rng = random.Random(seed)
    profiles = []
    seen_patients: set[str] = set()

    for sid in studies:
        recs = src.fetch_mutations(sid, gene_entrez)
        rep = run_qc(recs, sid)

        drop_keys = {(f.sample_id, f.check) for f in rep.findings if f.check in _DROP_CHECKS}
        flag_by_sample: dict[str, set[str]] = defaultdict(set)
        for f in rep.findings:
            if f.sample_id:
                flag_by_sample[f.sample_id].add(f.check)

        by_sample: dict[str, list[dict]] = defaultdict(list)
        seen_alt: set[tuple] = set()
        for r in recs:
            if (r["sample_id"], "duplicate_row") in drop_keys:
                # only skip the true duplicate occurrences, not the first
                key = (r["sample_id"], r["entrez_gene_id"], r["protein_change"])
                if key in seen_alt:
                    continue
                seen_alt.add(key)
            if (r["sample_id"], "vaf_zero_alt") in drop_keys and r.get("tumor_alt_count") == 0:
                continue
            if not r.get("protein_change"):
                continue
            by_sample[r["sample_id"]].append(r)

        # one representative sample per patient: the richest profile (most alterations),
        # tie-broken by sample id — deterministic, not shuffle-dependent.
        best_for_patient: dict[str, str] = {}
        for s, rs in by_sample.items():
            if len(rs) < min_alterations:
                continue
            pat = rs[0]["patient_id"]
            cur = best_for_patient.get(pat)
            if cur is None or (len(rs), s) > (len(by_sample[cur]), cur):
                best_for_patient[pat] = s

        # cap per study; shuffle only decides WHICH patients when capping (seeded)
        patients = list(best_for_patient)
        rng.shuffle(patients)
        taken = 0
        for pat in patients:
            if pat in seen_patients:      # cross-study dedup (same patient in another study)
                continue
            s = best_for_patient[pat]
            seen_patients.add(pat)
            profiles.append({
                "study_id": sid,
                "sample_id": s,
                "patient_id": pat,
                "tumor_type": _STUDY_TUMOR_TYPE.get(sid, "lung"),
                "alterations": by_sample[s],
                "qc_flags": sorted(flag_by_sample.get(s, set())),
            })
            taken += 1
            if taken >= per_study:
                break
    return profiles


# ---- step 5: annotation (evidence attachment, not labelling) ------------------------------
def annotate_alteration(gene, protein_change, tumor_type, civic, clinvar):
    """Attach CIViC (somatic) + ClinVar (germline) evidence, joined on canonical ids.

    Returns a plain dict of evidence — deliberately not a verdict. `actionable` means CIViC
    holds level-A/B predictive *sensitivity* evidence, a screening flag, not an ESCAT tier.
    """
    ev = civic.retrieve(EvidenceQuery(gene=gene, variant=protein_change,
                                      tumor_type=tumor_type, top_k=5))
    clinvar_ids = sorted({i for e in ev for i in e.payload.get("clinvar_ids", [])})
    caid = next((e.payload.get("caid") for e in ev if e.payload.get("caid")), None)
    germline = clinvar.by_ids(clinvar_ids) if clinvar_ids else []

    actionable = any(
        e.kind == EvidenceKind.PREDICTIVE
        and e.payload.get("evidence_direction") == "SUPPORTS"
        and "SENSITIVITY" in (e.payload.get("significance") or "")
        and (e.evidence_level in ("A", "B"))
        for e in ev
    )
    return {
        "caid": caid,                                   # canonical variant id (join key)
        "clinvar_ids": clinvar_ids,
        "actionable": actionable,
        "civic": [{"id": e.citation_id, "summary": e.summary} for e in ev],
        "clinvar": [{"id": c.citation_id, "summary": c.summary} for c in germline],
    }


# ---- step 6: assembly + patient-disjoint split -------------------------------------------
def assemble_case(profile, *, civic=None, clinvar=None) -> Case:
    """One profile -> one Case. Evidence (if connectors given) goes in provenance, not gold."""
    alts, evidence = [], {}
    for r in profile["alterations"]:
        gene, pc = r["gene_symbol"], r["protein_change"]
        alts.append(Alteration(
            gene=gene, variant=pc, kind="mutation",
            is_somatic=True,   # somatic mutation profile; the per-record status field is unusable
        ))
        if civic is not None and clinvar is not None:
            evidence[f"{gene} {pc}"] = annotate_alteration(
                gene, pc, profile["tumor_type"], civic, clinvar)

    return Case(
        case_id=f"{profile['study_id']}:{profile['sample_id']}",
        alterations=alts,
        context=ClinicalContext(tumor_type=profile["tumor_type"]),  # stage/lines/PS: honest None
        gold=None,  # unlabelled — gold recommendations are a Phase-3 labelling step
        provenance={
            "cbioportal_study": profile["study_id"],
            "sample_id": profile["sample_id"],
            "patient_id": profile["patient_id"],
            "build": "GRCh37",
            "qc_flags": profile["qc_flags"],
            "evidence": evidence,
        },
    )


def split_cases(cases, *, seed=17, fracs=(0.7, 0.1, 0.1, 0.1)) -> list[Case]:
    """Assign each case a Split. Cases are already one-per-patient, so a random split is
    patient-disjoint by construction (no leakage). Returns new Case objects with `split` set."""
    order = list(cases)
    random.Random(seed).shuffle(order)
    n = len(order)
    b1 = int(n * fracs[0]); b2 = b1 + int(n * fracs[1]); b3 = b2 + int(n * fracs[2])
    labelled = []
    for i, c in enumerate(order):
        split = (Split.TRAIN if i < b1 else Split.VAL if i < b2
                 else Split.TEST if i < b3 else Split.GENERALIZATION)
        labelled.append(Case(case_id=c.case_id, alterations=c.alterations, context=c.context,
                             gold=c.gold, split=split, provenance=c.provenance))
    return labelled
