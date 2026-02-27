"""Analyze evaluation results and produce CLI summary + enriched JSON."""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Stage classification ─────────────────────────────────────────────────

STAGE_CATEGORIES = [
    ("Presentation & Assessment", [
        r"initial", r"present", r"assess", r"admission", r"triage", r"history",
        r"complaint", r"arrival", r"emergency", r"referral", r"intake",
        r"clinical\s+eval", r"physical\s+exam", r"vital",
    ]),
    ("Diagnosis & Interpretation", [
        r"diagnos", r"workup", r"imaging", r"patholog", r"histopath",
        r"interpret", r"laboratory", r"biopsy", r"evaluation", r"test\b",
        r"molecular", r"immuno", r"cytolog", r"radiol", r"endoscop",
        r"analy[sz]", r"differential", r"investig", r"genetic",
        r"confirmatory", r"staging", r"grading", r"classif",
        r"acid.base", r"blood.gas", r"echocardiog", r"electrocardio",
        r"angiogra", r"biopsy", r"bronchoscop", r"colonoscop",
    ]),
    ("Therapeutic Strategy", [
        r"therap", r"treatment", r"management\s+plan", r"intervention",
        r"surgical", r"operat", r"resection", r"prescri", r"regimen",
        r"chemotherapy", r"radiation", r"procedur", r"decision",
        r"pharmacol", r"dosing", r"medication", r"induction", r"initiat",
        r"reconstruct", r"transplant", r"transfus", r"ventilat",
        r"anesthetic", r"sedation", r"reperfu", r"revascul",
        r"definitive\s+management", r"acute\s+management",
        r"medical\s+management", r"conservative",
    ]),
    ("Complication Management", [
        r"complicat", r"deteriorat", r"adverse", r"relapse", r"crisis",
        r"decompens", r"recurren", r"failure", r"toxic", r"reject",
        r"refractory", r"resistant", r"exacerbat", r"hemorrhag",
        r"infect", r"sepsis", r"graft.versus", r"side.effect",
        r"acute\s+(?:kidney|renal|liver|hepat|respiratory|neuro)",
    ]),
    ("Follow-up", [
        r"follow", r"monitor", r"surveil", r"outcome", r"prognos",
        r"long.term", r"recovery", r"rehabilit", r"discharge",
        r"month", r"year", r"week", r"counsel", r"screen",
        r"maintenance", r"recurrence\s+monitor",
    ]),
]

_STAGE_PATTERN_CACHE: List[Tuple[str, re.Pattern]] = []


def _get_stage_patterns() -> List[Tuple[str, re.Pattern]]:
    global _STAGE_PATTERN_CACHE
    if not _STAGE_PATTERN_CACHE:
        for name, keywords in STAGE_CATEGORIES:
            combined = "|".join(keywords)
            _STAGE_PATTERN_CACHE.append((name, re.compile(combined, re.IGNORECASE)))
    return _STAGE_PATTERN_CACHE


def classify_stage(stage_name: str) -> str:
    """Map a granular stage name to one of 5 broad clinical categories."""
    if not stage_name:
        return "Other"
    for category, pattern in _get_stage_patterns():
        if pattern.search(stage_name):
            return category
    return "Other"


# ── Modality extraction ──────────────────────────────────────────────────

PRIMARY_IMAGE_MODALITIES = [
    "X-ray", "CT", "MRI", "Ultrasound", "Clinical Photo",
    "Endoscopy", "Angiography", "PET & SPECT",
    "Pathology", "Biosignals", "Diagram & Plot",
]

_IMAGE_MODALITY_RULES = [
    ("X-ray", [r"\bX-ray\b", r"\bRadiography\b"]),
    ("CT", [r"\bCT\b", r"\bComputed Tomography\b"]),
    ("MRI", [r"\bMRI\b", r"\bMagnetic Resonance\b"]),
    ("Ultrasound", [r"\bUltrasound\b", r"\bEchocardiogram\b", r"\bDoppler\b"]),
    ("Clinical Photo", [r"\bClinical Photograph\b", r"\bDermoscopy\b",
                        r"\bFundus Photograph\b", r"\bRetinography\b",
                        r"\bColposcopy\b", r"\bCystoscopy\b"]),
    ("Endoscopy", [r"\bEndoscopy\b", r"\bColonoscopy\b", r"\bBronchoscopy\b",
                   r"\bLaparoscopy\b", r"\bHysteroscopy\b"]),
    ("Angiography", [r"\bAngiography\b", r"\bAngiogram\b", r"\bFluoroscopy\b"]),
    ("PET & SPECT", [r"\bPET\b", r"\bSPECT\b", r"\bNuclear Medicine\b",
                     r"\bBone Scan\b", r"\bScintigraphy\b"]),
    ("Pathology", [r"\bHistopathology\b", r"\bGross Pathology\b", r"\bCytolog\b"]),
    ("Biosignals", [r"\bECG\b", r"\bEEG\b", r"\bElectrocardiog\b",
                    r"\bElectroencephalog\b", r"\bPhysiological Signal\b",
                    r"\bElectrocochleogram\b", r"\bElectroretinogram\b"]),
    ("Diagram & Plot", [r"\bDiagram\b", r"\bIllustration\b", r"\bStatistical Plot\b",
                        r"\bFlowchart\b"]),
]

_IMAGE_MOD_PATTERNS: List[Tuple[str, re.Pattern]] = []


def _get_image_patterns() -> List[Tuple[str, re.Pattern]]:
    global _IMAGE_MOD_PATTERNS
    if not _IMAGE_MOD_PATTERNS:
        for name, rules in _IMAGE_MODALITY_RULES:
            combined = "|".join(rules)
            _IMAGE_MOD_PATTERNS.append((name, re.compile(combined, re.IGNORECASE)))
    return _IMAGE_MOD_PATTERNS


def extract_image_modalities(type_str: str) -> Set[str]:
    """Extract primary image modalities from a compound type string."""
    mods: Set[str] = set()
    for name, pattern in _get_image_patterns():
        if pattern.search(type_str):
            mods.add(name)
    return mods


PRIMARY_TABLE_MODALITIES = [
    "Lab Results", "Medications", "Demographics",
    "Monitoring", "Literature", "Genomics",
    "Pathology & IHC", "Procedures", "Staging Sys",
]

_TABLE_MODALITY_RULES = [
    ("Lab Results", [r"\bLaboratory\b", r"\bLab\b", r"\bBlood\b",
                     r"\bSerum\b", r"\bBiomarker\b", r"\bRenal Function\b",
                     r"\bLiver Function\b", r"\bTumor Marker\b",
                     r"\bVital Sign\b", r"\bAcid.Base\b",
                     r"\bOrgan Measurement\b", r"\bImmun\b"]),
    ("Medications", [r"\bMedication\b", r"\bDrug\b", r"\bDos\b",
                     r"\bTreatment Regimen\b", r"\bPharmaco\b",
                     r"\bNutrition\b", r"\bTreatment Plan\b",
                     r"\bChemotherapy\b"]),
    ("Demographics", [r"\bDemograph\b", r"\bBaseline\b", r"\bPatient\b",
                      r"\bAge\b", r"\bGrowth\b"]),
    ("Monitoring", [r"\bMonitor\b", r"\bLongitudinal\b", r"\bFollow\b",
                    r"\bSurvival\b", r"\bOutcome\b", r"\bResponse\b",
                    r"\bPre.Post\b", r"\bClinical Course\b",
                    r"\bRecovery\b", r"\bProgression\b"]),
    ("Literature", [r"\bLiterature\b", r"\bCase Series\b", r"\bReview\b",
                    r"\bComparison\b", r"\bGuideline\b"]),
    ("Genomics", [r"\bGenom", r"\bGenetic", r"\bMutation",
                  r"\bPrimer", r"\bSequenc", r"\bMolecular"]),
    ("Pathology & IHC", [r"\bPathology\b", r"\bImmunohistochem\b",
                         r"\bHistolog\b", r"\bIHC\b"]),
    ("Procedures", [r"\bProcedur\b", r"\bSurgical\b", r"\bOperati\b",
                    r"\bTransplant\b", r"\bIntervention\b",
                    r"\bEngraftment\b", r"\bDosimetric\b"]),
    ("Staging Sys", [r"\bStaging\b", r"\bClassification\b", r"\bScoring\b",
                     r"\bGrading\b", r"\bRisk\b", r"\bPrognost\b",
                     r"\bSeverity\b", r"\bTumor\b", r"\bTNM\b"]),
]

_TABLE_MOD_PATTERNS: List[Tuple[str, re.Pattern]] = []


def _get_table_patterns() -> List[Tuple[str, re.Pattern]]:
    global _TABLE_MOD_PATTERNS
    if not _TABLE_MOD_PATTERNS:
        for name, rules in _TABLE_MODALITY_RULES:
            combined = "|".join(rules)
            _TABLE_MOD_PATTERNS.append((name, re.compile(combined, re.IGNORECASE)))
    return _TABLE_MOD_PATTERNS


def extract_table_modalities(type_str: str) -> Set[str]:
    """Extract primary table modalities from a compound type string."""
    mods: Set[str] = set()
    for name, pattern in _get_table_patterns():
        if pattern.search(type_str):
            mods.add(name)
    return mods


# ── Core analysis ────────────────────────────────────────────────────────

def _acc(correct: int, total: int) -> float:
    return round(correct / total, 4) if total > 0 else 0.0


def _block(s: Dict[str, int]) -> Dict[str, Any]:
    return {
        "cases": s.get("cases", 0),
        "correct_cases": s.get("correct_cases", 0),
        "case_accuracy": _acc(s.get("correct_cases", 0), s.get("cases", 0)),
        "mcqs": s.get("mcqs", 0),
        "correct_mcqs": s.get("correct_mcqs", 0),
        "question_accuracy": _acc(s.get("correct_mcqs", 0), s.get("mcqs", 0)),
    }


def _mcq_block(s: Dict[str, int]) -> Dict[str, Any]:
    return {
        "mcqs": s["mcqs"],
        "correct_mcqs": s["correct_mcqs"],
        "question_accuracy": _acc(s["correct_mcqs"], s["mcqs"]),
    }


def analyze(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive summary statistics from evaluated results.
    Writes results["summary"] and returns the mutated dict.
    """
    cases = results.get("cases", [])
    if not cases:
        results["summary"] = {}
        return results

    total_cases = len(cases)
    total_mcqs = 0
    correct_mcqs = 0
    correct_cases = 0

    rarity_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "correct_cases": 0, "mcqs": 0, "correct_mcqs": 0}
    )
    chapter_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "correct_cases": 0, "mcqs": 0, "correct_mcqs": 0}
    )
    stage_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"mcqs": 0, "correct_mcqs": 0}
    )
    position_stats: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"mcqs": 0, "correct_mcqs": 0, "errors": 0}
    )
    image_mod_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"mcqs": 0, "correct_mcqs": 0}
    )
    table_mod_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"mcqs": 0, "correct_mcqs": 0}
    )

    for case in cases:
        mcqs = case.get("mcqs", [])
        n = len(mcqs)
        c = case.get("correct_mcqs", sum(1 for m in mcqs if m.get("is_correct")))
        total_mcqs += n
        correct_mcqs += c

        case_correct = (c == n and n > 0)
        if case_correct:
            correct_cases += 1

        # By rarity
        rarity = str(case.get("rarity", "unknown")).lower()
        r = rarity_stats[rarity]
        r["cases"] += 1
        r["correct_cases"] += int(case_correct)
        r["mcqs"] += n
        r["correct_mcqs"] += c

        # By ICD-10 chapter
        chapter = case.get("level1", "Unknown")
        ch = chapter_stats[chapter]
        ch["cases"] += 1
        ch["correct_cases"] += int(case_correct)
        ch["mcqs"] += n
        ch["correct_mcqs"] += c

        # Per-MCQ analysis
        for m in mcqs:
            is_correct = m.get("is_correct", False)
            has_error = bool(m.get("error"))

            # By clinical stage (broad category)
            stage_cat = classify_stage(m.get("stage", ""))
            ss = stage_stats[stage_cat]
            ss["mcqs"] += 1
            if is_correct:
                ss["correct_mcqs"] += 1

            # By MCQ position (0-indexed)
            pos = m.get("mcq_index", 0)
            ps = position_stats[pos]
            ps["mcqs"] += 1
            if is_correct:
                ps["correct_mcqs"] += 1
            if has_error:
                ps["errors"] += 1

            # By image modality
            for type_str in m.get("image_types", []):
                for mod in extract_image_modalities(type_str):
                    ims = image_mod_stats[mod]
                    ims["mcqs"] += 1
                    if is_correct:
                        ims["correct_mcqs"] += 1

            # By table modality
            for type_str in m.get("table_types", []):
                for mod in extract_table_modalities(type_str):
                    ts = table_mod_stats[mod]
                    ts["mcqs"] += 1
                    if is_correct:
                        ts["correct_mcqs"] += 1

    # Build summary
    stage_order = [cat for cat, _ in STAGE_CATEGORIES] + ["Other"]
    by_stage = {}
    for cat in stage_order:
        if cat in stage_stats:
            by_stage[cat] = _mcq_block(stage_stats[cat])

    by_position = {}
    for pos in sorted(position_stats.keys()):
        ps = position_stats[pos]
        by_position[f"Q{pos + 1}"] = {
            "mcqs": ps["mcqs"],
            "correct_mcqs": ps["correct_mcqs"],
            "question_accuracy": _acc(ps["correct_mcqs"], ps["mcqs"]),
            "errors": ps["errors"],
            "error_rate": _acc(ps["errors"], ps["mcqs"]),
        }

    by_image = {
        k: _mcq_block(v)
        for k, v in sorted(image_mod_stats.items(), key=lambda x: -x[1]["mcqs"])
    }

    by_table = {
        k: _mcq_block(v)
        for k, v in sorted(table_mod_stats.items(), key=lambda x: -x[1]["mcqs"])
    }

    results["summary"] = {
        "total_cases": total_cases,
        "total_mcqs": total_mcqs,
        "correct_mcqs": correct_mcqs,
        "correct_cases": correct_cases,
        "question_accuracy": _acc(correct_mcqs, total_mcqs),
        "case_accuracy": _acc(correct_cases, total_cases),
        "by_rarity": {k: _block(v) for k, v in sorted(rarity_stats.items())},
        "by_chapter": {
            k: _block(v)
            for k, v in sorted(chapter_stats.items(), key=lambda x: -x[1]["mcqs"])
        },
        "by_stage": by_stage,
        "by_position": by_position,
        "by_image_modality": by_image,
        "by_table_modality": by_table,
    }

    return results


# ── CLI summary printer ─────────────────────────────────────────────────

def print_summary(results: Dict[str, Any]) -> None:
    """Print a concise evaluation summary to the terminal."""
    meta = results.get("meta", {})
    s = results.get("summary", {})

    if not s:
        print("No results to summarize.")
        return

    model = meta.get("model", "?")
    dataset = meta.get("dataset", "?")

    w = 60
    print()
    print("=" * w)
    print(f"  LiveClin Results: {model} ({dataset})")
    print("=" * w)

    total_q = s.get("total_mcqs", 0)
    correct_q = s.get("correct_mcqs", 0)
    total_c = s.get("total_cases", 0)
    correct_c = s.get("correct_cases", 0)
    q_acc = s.get("question_accuracy", 0)
    c_acc = s.get("case_accuracy", 0)

    print(f"  Question Accuracy:  {correct_q}/{total_q} ({q_acc:.1%})")
    print(f"  Case Accuracy:      {correct_c}/{total_c} ({c_acc:.1%})")
    print("-" * w)

    by_rarity = s.get("by_rarity", {})
    if by_rarity:
        print("  By Rarity:")
        for rarity, rs in by_rarity.items():
            label = rarity.capitalize()
            n = rs["cases"]
            qa = rs["question_accuracy"]
            ca = rs["case_accuracy"]
            print(f"    {label:<12} ({n:>4} cases)  Q-Acc {qa:.1%}  C-Acc {ca:.1%}")
        print("-" * w)

    by_stage = s.get("by_stage", {})
    if by_stage:
        print("  By Clinical Stage:")
        for stage, ss in by_stage.items():
            n = ss["mcqs"]
            qa = ss["question_accuracy"]
            print(f"    {stage:<30} ({n:>4} MCQs)  Q-Acc {qa:.1%}")
        print("-" * w)

    by_position = s.get("by_position", {})
    if by_position:
        print("  By Question Position:")
        for pos, ps in by_position.items():
            n = ps["mcqs"]
            qa = ps["question_accuracy"]
            err = ps["error_rate"]
            print(f"    {pos:<6} ({n:>4} MCQs)  Q-Acc {qa:.1%}  Err {err:.1%}")
        print("-" * w)

    by_chapter = s.get("by_chapter", {})
    if by_chapter:
        print("  By Chapter (top 5):")
        for i, (ch, cs) in enumerate(by_chapter.items()):
            if i >= 5:
                break
            short = ch[:40] + ("..." if len(ch) > 40 else "")
            n = cs["cases"]
            qa = cs["question_accuracy"]
            print(f"    {short:<44} ({n:>3}) Q-Acc {qa:.1%}")
        if len(by_chapter) > 5:
            print(f"    ... and {len(by_chapter) - 5} more chapters")
        print("-" * w)

    by_image = s.get("by_image_modality", {})
    if by_image:
        print("  By Image Modality:")
        for mod, ms in by_image.items():
            n = ms["mcqs"]
            qa = ms["question_accuracy"]
            print(f"    {mod:<20} ({n:>4} MCQs)  Q-Acc {qa:.1%}")
        print("-" * w)

    by_table = s.get("by_table_modality", {})
    if by_table:
        print("  By Table Modality:")
        for mod, ms in by_table.items():
            n = ms["mcqs"]
            qa = ms["question_accuracy"]
            print(f"    {mod:<20} ({n:>4} MCQs)  Q-Acc {qa:.1%}")
        print("-" * w)

    print(f"  Results: {meta.get('image_mode', '?')} mode | "
          f"v{meta.get('liveclin_version', '?')}")
    print("=" * w)
    print()
