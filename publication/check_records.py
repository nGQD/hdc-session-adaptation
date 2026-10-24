"""Validate the published analyses without recordings, checkpoints, or a GPU."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "ieee_access/experiments"
RESULTS = EXPERIMENTS / "results"
sys.path.insert(0, str(EXPERIMENTS))

from analyze_revision import verify_analysis
from baseline_robustness_audit import audit
from measure_resources import validate_recorded_resources


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    analysis = verify_analysis(RESULTS / "revised_study",
                               RESULTS / "revised_endpoint/endpoint_analysis.json")
    baseline = audit(RESULTS / "baseline_robustness", require_complete=True)
    if baseline != read(RESULTS / "baseline_robustness/audit.json") or not baseline["passed"]:
        raise RuntimeError("Baseline records differ from their completed audit")
    resources = validate_recorded_resources()
    if len(resources) != 24 or sum(len(r["raw_records_sha256"]) for r in resources) != 72:
        raise RuntimeError("Resource measurement coverage is incomplete")
    for folder, manifest_name, hashes_key in [
        ("feature_controls", "provenance.json", "result_sha256"),
        ("confirmed_sensitivity", "audit.json", "artifact_sha256"),
    ]:
        manifest = read(RESULTS / folder / manifest_name)
        for relative, expected in manifest[hashes_key].items():
            path = RESULTS / folder / relative
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f"Missing or changed result: {path.relative_to(ROOT)}")
    print(json.dumps({
        "passed": True,
        "stream_records": len(analysis["inputs"]),
        "statistics_outputs": len(analysis["statistics"]),
        "baseline_counts": baseline["counts"],
        "resource_groups": len(resources),
        "resource_repetitions": 72,
        "feature_and_sensitivity_record_hashes": "verified",
    }, indent=2))


if __name__ == "__main__":
    main()
