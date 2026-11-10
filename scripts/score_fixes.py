"""Read the five fix sweeps on both loops under one rule, and check the reference arms.

    python scripts/score_fixes.py

Each sweep script prints its own table and judges nothing. This reads what they
wrote, for the loop as it ships (outputs/) and the loop as it first shipped
(outputs/first_shipped/), and applies the rule registered before the re-run in
docs/registrations/2026-11-10-fixes-on-the-loop-as-shipped.md:

    a setting counts for or against a fix only where it clears zero, in the
    same direction, in two cities.

One arm-and-city pair clearing zero is not a result. Close to a hundred
intervals are read here and none is corrected for the others, so a handful would
clear zero if no fix did anything.

It also checks the instrument. On the loop as it ships, each sweep's reference
arm is the shipped loop itself, so its retraining premium has to match the
six-city table to one decimal. A sweep that fails that for a city is void there.

Writes outputs/fix_verdicts.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
LOOPS = {"shipped": OUTPUTS, "first": OUTPUTS / "first_shipped"}

#: file, the column naming the arm, the reference arm, and the comparison columns
#: (point, flag) read over the weeks the arm served a different model.
SWEEPS = {
    "second retrain alarm": ("skill_floor_sweep.csv", "floor", "off", "vs_off_acted"),
    "longer exam": ("holdout_sweep.csv", "holdout", "7", "vs_7d"),
    "re-certification": ("recertify_sweep.csv", "cadence", "off", "vs_off_acted"),
    "confidence-aware gate": ("promotion_confidence_sweep.csv", "confidence", "off", "vs_off_acted"),
    "rollback": ("probation_sweep.csv", "probation", "off", "vs_off_acted"),
}
#: The arm of each sweep that is the shipped loop, whose premium must reproduce.
SHIPPED_ARM = {"second retrain alarm": "-0.50", "longer exam": "7"}
TWO_CITIES = 2


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cleared(row: dict[str, str], column: str) -> int:
    """+1 or -1 where the interval excludes zero, 0 where it covers it or is absent."""
    if not row.get(column) or row.get(f"{column}_real") != "True":
        return 0
    return 1 if float(row[column]) > 0 else -1


def read_sweep(folder: Path, name: str) -> dict:
    file, arm_col, reference, column = SWEEPS[name]
    table = rows(folder / file)
    settings: dict[str, dict] = {}
    for row in table:
        arm = row[arm_col]
        if arm == reference:
            continue
        entry = settings.setdefault(arm, {"intervals": 0, "helped": [], "harmed": []})
        if row.get(column):
            entry["intervals"] += 1
        sign = cleared(row, column)
        if sign:
            entry["helped" if sign > 0 else "harmed"].append(row["city"])
    for entry in settings.values():
        entry["verdict"] = (
            "pays" if len(entry["helped"]) >= TWO_CITIES
            else "harmful" if len(entry["harmed"]) >= TWO_CITIES
            else "nothing measurable"
        )
    verdicts = {e["verdict"] for e in settings.values()}
    return {
        "settings": settings,
        "intervals": sum(e["intervals"] for e in settings.values()),
        "helped": sum(len(e["helped"]) for e in settings.values()),
        "harmed": sum(len(e["harmed"]) for e in settings.values()),
        # A fix can pay at one setting and harm at another, and the second alarm did.
        "verdict": "pays or harms, by setting" if {"pays", "harmful"} <= verdicts
        else "pays" if "pays" in verdicts else "harmful" if "harmful" in verdicts
        else "nothing measurable",
    }


def reference_check(name: str) -> list[dict]:
    """The shipped-loop arm of each sweep against the published six-city premiums."""
    file, arm_col, reference, _ = SWEEPS[name]
    arm = SHIPPED_ARM.get(name, reference)
    published = {
        city["label"]: city["interval"]["when_it_acted"]["point"]
        for city in json.loads((OUTPUTS / "uncertainty.json").read_text(encoding="utf-8"))["cities"]
    }
    out = []
    for row in rows(OUTPUTS / file):
        if row[arm_col] != arm:
            continue
        got, want = round(float(row["paired"]), 1), round(published[row["city"]], 1)
        out.append({"city": row["city"], "arm": got, "published": want,
                    "passed": abs(got - want) <= 0.1})
    return out


def main() -> int:
    report: dict = {"rule": f"clears zero, same direction, in {TWO_CITIES} cities", "loops": {}}
    void = False
    for loop, folder in LOOPS.items():
        print(f"\n=== the loop as {'it ships' if loop == 'shipped' else 'first shipped'} ===")
        report["loops"][loop] = {}
        total = helped = harmed = 0
        for name in SWEEPS:
            if not (folder / SWEEPS[name][0]).exists():
                print(f"  {name:<24} not run")
                continue
            result = read_sweep(folder, name)
            report["loops"][loop][name] = result
            total, helped, harmed = (
                total + result["intervals"], helped + result["helped"], harmed + result["harmed"]
            )
            print(f"  {name:<24} {result['verdict']:<19} {result['intervals']:>2} intervals, "
                  f"{result['helped']} clear zero as a gain, {result['harmed']} as a harm")
            for arm, entry in result["settings"].items():
                if entry["helped"] or entry["harmed"]:
                    print(f"      {arm:>6}: gain in {entry['helped'] or '-'}, "
                          f"harm in {entry['harmed'] or '-'}")
        report["loops"][loop]["_total"] = {"intervals": total, "helped": helped, "harmed": harmed}
        print(f"  {total} intervals in all: {helped} clear zero as a gain, {harmed} as a harm")

    print("\n=== reference arms on the loop as it ships ===")
    report["reference"] = {}
    for name in SWEEPS:
        if not (OUTPUTS / SWEEPS[name][0]).exists():
            continue
        checks = reference_check(name)
        report["reference"][name] = checks
        failed = [c for c in checks if not c["passed"]]
        void = void or bool(failed) or len(checks) != 6
        print(f"  {name:<24} {len(checks) - len(failed)} of {len(checks)} cities reproduce"
              + "".join(f"\n      VOID {c['city']}: arm {c['arm']:+.1f}%, published "
                        f"{c['published']:+.1f}%" for c in failed))

    (OUTPUTS / "fix_verdicts.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUTPUTS / 'fix_verdicts.json'}")
    return 1 if void else 0


if __name__ == "__main__":
    sys.exit(main())
