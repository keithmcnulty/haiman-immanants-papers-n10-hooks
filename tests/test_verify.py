"""The verification runner (monimm/verify.py): the stored tier passes on the repository's
results, and a corrupted result file or checksum is reported as a failure."""
import json
import shutil

import pytest

import monimm.verify as V


def test_stored_tier_passes():
    ver = V.Verifier(V.ROOT, verbose=False)
    ver.stored()
    failed = [L for L in ver.lines if L.status == "fail"]
    assert not failed, failed
    assert {L.check for L in ver.lines} >= {"1", "2", "3", "4", "5", "6", "7", "8", "10", "11", "12", "13", "paper", "sha"}


@pytest.fixture
def copy_of_results(tmp_path, monkeypatch):
    res = tmp_path / "results"
    res.mkdir()
    for p in V.RESULTS.glob("*.json"):
        shutil.copy(p, res)
    shutil.copy(V.SUMS, res)
    monkeypatch.setattr(V, "ROOT", tmp_path)
    monkeypatch.setattr(V, "RESULTS", res)
    monkeypatch.setattr(V, "SUMS", res / "SHA256SUMS")
    monkeypatch.setattr(V, "git_tracked", lambda: [])   # files not copied count as absent, not failed
    return res


def run_stored(tmp_path):
    ver = V.Verifier(tmp_path, verbose=False)
    ver.stored()
    return ver.lines


def test_copy_passes(copy_of_results, tmp_path):
    assert not [L for L in run_stored(tmp_path) if L.status == "fail"]


def test_corrupted_flag_fails(copy_of_results, tmp_path):
    p = copy_of_results / "kl_graded_n10.json"
    d = json.loads(p.read_text())
    d["non_unimodal"] = 1
    p.write_text(json.dumps(d, indent=1))
    failed = {(L.check, L.n) for L in run_stored(tmp_path) if L.status == "fail"}
    assert ("Thm", 10) in failed and ("sha", None) in failed


def test_paper_number_mismatch_fails(copy_of_results, tmp_path):
    p = copy_of_results / "coxeter3_check_n10.json"
    d = json.loads(p.read_text())
    d["extremal_pairs_checked"] -= 1
    p.write_text(json.dumps(d, indent=1))
    failed = {(L.check, L.n) for L in run_stored(tmp_path) if L.status == "fail"}
    assert ("2", 10) in failed and ("paper", 10) in failed
