"""Shared, offline bookkeeping for the SpecCon / author MT-Bench protocol."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

AUTHOR_COMMIT = "6666d8bdb37ebd153a8916b0464c8a3878976aae"
AUTHOR_ROOT = Path("/workspace/DPLoRA")
AUTHOR_PYTHON = AUTHOR_ROOT / ".venv-author/bin/python"
BASE_SHA256 = "68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_environment(requirements):
    """Compare every pinned distribution, including CUDA local version suffixes."""
    from packaging.version import Version
    versions, mismatches = {}, []
    for line in Path(requirements).read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line.strip())
        if not match:
            continue
        name, expected = match.groups()
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "MISSING"
        versions[name] = actual
        if actual == "MISSING" or Version(actual) != Version(expected):
            mismatches.append(f"{name}: {actual} != {expected}")
    if not versions:
        raise RuntimeError("No pinned requirements found")
    if mismatches:
        raise RuntimeError("Author dependency mismatch: " + "; ".join(mismatches))
    if sys.version_info[:3] != (3, 10, 12):
        raise RuntimeError("Author environment requires Python 3.10.12")
    return versions


def verify_snapshot(output):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    for name, digest in manifest["snapshot_sha256"].items():
        if sha256(output / name) != digest:
            raise RuntimeError("Frozen source/config changed: " + name)
    return manifest


def validate_answers(answer_file, question_file, model_id):
    questions = rows(question_file)
    answers = rows(answer_file)
    expected = {q["question_id"] for q in questions}
    if len(expected) != 80 or len(answers) != 80 or {a["question_id"] for a in answers} != expected:
        raise RuntimeError("MT-Bench must contain exactly 80 unique official questions")
    for answer in answers:
        choices = answer.get("choices", [])
        if answer.get("model_id") != model_id or len(choices) != 1 or choices[0].get("index") != 0:
            raise RuntimeError("Wrong answer model ID or candidate count")
        turns = choices[0].get("turns", [])
        if len(turns) != 2 or any(not isinstance(t, str) or not t.strip() or t == "ERROR" for t in turns):
            raise RuntimeError("Missing/failed MT-Bench turn")
    return {a["question_id"]: a for a in answers}


def summarize_judgments(path, question_file, model_id):
    questions = {q["question_id"]: q for q in rows(question_file)}
    judgments = rows(path)
    keys = [(r["question_id"], r["turn"]) for r in judgments]
    expected = {(qid, turn) for qid in questions for turn in (1, 2)}
    if len(judgments) != 160 or set(keys) != expected:
        raise RuntimeError("Require all 160 unique judgments; refusing a partial/duplicate mean")
    for row in judgments:
        ref = questions[row["question_id"]]["category"] in {"math", "reasoning", "coding"}
        template = "single-math-v1" if ref else "single-v1"
        if row["turn"] == 2:
            template += "-multi-turn"
        if row["model"] != model_id or row["judge"] != ["gpt-4", template]:
            raise RuntimeError("Unexpected judge, rubric or model")
        if not 1 <= row["score"] <= 10:
            raise RuntimeError("Invalid rating; refusing to hide failures in the mean")
    mean = lambda selected: sum(r["score"] for r in selected) / len(selected)
    return {
        "model_id": model_id, "judge_model": "gpt-4", "judgments": 160,
        "score": mean(judgments),
        "first_turn": mean([r for r in judgments if r["turn"] == 1]),
        "second_turn": mean([r for r in judgments if r["turn"] == 2]),
        "categories": {cat: mean([r for r in judgments if questions[r["question_id"]]["category"] == cat])
                       for cat in sorted({q["category"] for q in questions.values()})},
    }
