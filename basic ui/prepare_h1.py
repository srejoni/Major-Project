#!/usr/bin/env python3
"""I1 helper: build the H1 package (confirmation + scripts) for the LLM owner.

Read-only on the repo and on the 297 handoffs; writes only under --out.
Nothing is regenerated, repaired or edited. Point --dir297 at the 297 folder,
never at the Phase-G 20.

Exit codes: 0 package written | 1 a check failed (nothing written) | 2 bad arguments.

Example:
  python prepare_h1.py --repo "/path/to/PDSCD" --dir297 "/path/to/297_studies" --out "H1_package"
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

DEFAULT_COMMIT = "e5e7432"
PHASE_G_VERSION = "1.0.0+dbe03c1"
MODEL_IDS = (
    "convnext_tiny_3038c523_20260925T164416Z",
    "efficientnet_b3_d5f1cbb3_20260918T175038Z",
    "resnet34_e849b5f0_20260925T175010Z",
)
GT_SWAP = [2797118205]
N_EXPECTED = 297
SEND = ("export_phase8_batch.py", "assemble_study_output.py", "pdscd_output_schema.json")
KEY_RE = re.compile(
    r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gsk_[A-Za-z0-9]{20,}"
    r"|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
    r"|(?i:api[_-]?key|token|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"])"
)

results = []


def check(ok, label, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + label + (f"   [{detail}]" if detail else ""))
    return ok


def info(label, detail=""):
    print("INFO  " + label + (f"   [{detail}]" if detail else ""))


def git(repo, *args, must=True):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if must and p.returncode != 0:
        sys.exit("git " + " ".join(args) + " failed: " + p.stderr.decode("utf-8", "replace").strip())
    return p


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="PDSCD git repo (needs full history, not a shallow clone)")
    ap.add_argument("--dir297", required=True, help="folder holding the 297 study_*.json + handoff_manifest.json")
    ap.add_argument("--out", default="H1_package", help="output folder (must not exist yet)")
    ap.add_argument("--commit", default=DEFAULT_COMMIT, help="writer commit stamped in the files")
    a = ap.parse_args()

    repo, d297, out = Path(a.repo), Path(a.dir297), Path(a.out)
    pipeline = f"1.0.0+{a.commit[:7]}"
    if not (repo / ".git").exists():
        print(f"--repo is not a git repo: {repo}")
        return 2
    if not d297.is_dir():
        print(f"--dir297 is not a folder: {d297}")
        return 2
    out_r, d297_r = out.resolve(), d297.resolve()
    if out_r == d297_r or d297_r in out_r.parents:
        print("--out must not be inside the 297 folder")
        return 2
    if out.exists():
        print(f"--out already exists, remove it or choose another: {out}")
        return 2

    # ---- A. repo: the commit, and the three files as committed there
    print("== A. repo")
    r = git(repo, "rev-parse", "--verify", a.commit + "^{commit}", must=False)
    if not check(r.returncode == 0, f"commit {a.commit} exists",
                 "if the clone is shallow run: git fetch --unshallow" if r.returncode else ""):
        return 1
    full = r.stdout.decode().strip()
    head = git(repo, "rev-parse", "--short", "HEAD").stdout.decode().strip()
    anc = git(repo, "merge-base", "--is-ancestor", full, "HEAD", must=False).returncode == 0
    check(anc, f"{a.commit} is an ancestor of HEAD", f"HEAD {head}")
    st = git(repo, "status", "--porcelain", must=False)  # informational only; may fail if git-lfs is absent
    if st.returncode != 0:
        info("working tree status unavailable (ignored: files are read from the commit)")
    else:
        dirty = st.stdout.decode().strip()
        info("working tree " + ("has uncommitted changes (ignored: files are read from the commit)" if dirty else "clean"))

    tree = git(repo, "ls-tree", "-r", "--name-only", full).stdout.decode("utf-8").splitlines()
    sent, blobs = {}, {}
    for name in SEND:
        hits = [t for t in tree if Path(t).name == name]
        if not check(len(hits) == 1, f"{name} found exactly once at {a.commit}", ", ".join(hits) or "none"):
            continue
        path = hits[0]
        data = git(repo, "show", f"{full}:{path}").stdout
        check(not data.startswith(b"version https://git-lfs"), f"{name} is real content, not an LFS pointer")
        check(KEY_RE.search(data.decode("utf-8", "replace")) is None,
              f"{name} has no key-like strings", "review by hand if this fails")
        blob_c = git(repo, "rev-parse", f"{full}:{path}").stdout.decode().strip()
        h = git(repo, "rev-parse", f"HEAD:{path}", must=False)
        if h.returncode != 0:
            vs = "absent at HEAD"
        else:
            vs = "same as HEAD" if h.stdout.decode().strip() == blob_c else "differs from HEAD"
        info(f"{name}: {path}", f"blob {blob_c[:8]}, {vs}")
        sent[name] = {"repo_path": path, "git_blob": blob_c, "vs_head": vs,
                      "sha256": sha256_bytes(data), "bytes": len(data)}
        blobs[name] = data

    # ---- B. the 297 handoffs: read only
    print("== B. the 297 handoffs (read-only)")
    files = sorted(d297.glob("study_*.json"))
    check(len(files) == N_EXPECTED, f"study_*.json count == {N_EXPECTED}", str(len(files)))
    pv, sv, mv, em, ids, bad = Counter(), Counter(), Counter(), Counter(), [], []
    for p in files:
        try:
            o = json.loads(p.read_text(encoding="utf-8"))
            pv[o["pipeline_version"]] += 1
            sv[o["schema_version"]] += 1
            mv[json.dumps(o["model_versions"], sort_keys=True, ensure_ascii=False)] += 1
            em[json.dumps(o["ensemble_method"], sort_keys=True, ensure_ascii=False)] += 1
            ids.append(o["study_id"])
        except Exception as e:  # report, never repair
            bad.append(f"{p.name}: {type(e).__name__}")
    check(not bad, "every file read and has the fields this script needs", "; ".join(bad[:5]))
    n = len(files) - len(bad)
    check(dict(pv) == {pipeline: n}, f"pipeline_version is {pipeline} in all files", str(dict(pv)))
    check(PHASE_G_VERSION not in pv, f"no Phase-G version ({PHASE_G_VERSION}) mixed in")
    check(dict(sv) == {"1.0.0": n}, "schema_version is 1.0.0 in all files", str(dict(sv)))
    check(len(mv) == 1, "one distinct model_versions block", f"{len(mv)} distinct")
    mv_json = next(iter(mv)) if mv else ""
    check(all(m in mv_json for m in MODEL_IDS), "the 3 expected checkpoint ids appear in it")
    check(len(em) == 1, "one distinct ensemble_method block", f"{len(em)} distinct")
    em_json = next(iter(em)) if em else ""
    check(len(set(ids)) == len(ids), "study_id values are unique")
    check(GT_SWAP[0] in ids, f"GT-swap study {GT_SWAP[0]} is in the set")

    mp = d297 / "handoff_manifest.json"
    man = {}
    if check(mp.exists(), "handoff_manifest.json present"):
        man = json.loads(mp.read_text(encoding="utf-8"))
        check(man.get("n_studies") == len(files), "manifest n_studies equals file count", str(man.get("n_studies")))
        check(man.get("known_gt_swap_ids_in_set") == GT_SWAP, "manifest GT-swap list is exactly the expected id",
              str(man.get("known_gt_swap_ids_in_set")))
        check(set(man.get("study_ids", [])) == set(ids), "manifest study_ids equal the ids in the files")

    if not all(results):
        print("\nAt least one check failed. Nothing written. Do not edit or regenerate any handoff file;"
              " investigate, and tell the LLM owner if a file is affected.")
        return 1

    # ---- C. write the package
    print("== C. package")
    out.mkdir(parents=True)
    for name, data in blobs.items():
        (out / name).write_bytes(data)
    sums = "".join(f"{v['sha256']}  {k}\n" for k, v in sent.items())
    (out / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")

    facts = {"commit": full, "pipeline_version": pipeline, "n_studies": n,
             "model_versions": json.loads(mv_json), "ensemble_method": json.loads(em_json),
             "gt_swap_ids_in_set": GT_SWAP, "manifest_ids_sha256": man.get("ids_sha256"),
             "sent_files": sent}
    (out / "H1_facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")

    notes = "".join(f"  - `{k}` differs from HEAD; the copy sent is the one that produced the 297.\n"
                    for k, v in sent.items() if v["vs_head"] != "same as HEAD")
    msg = f"""**Subject: H1 - PDSCD 297-study handoff: confirmation and scripts**

Confirming that the {n} study files you already hold are final. Everything below was read from the files and the repo today, not from memory.

**The handoff**
- {n} `study_<id>.json` files, `schema_version` 1.0.0. `handoff_manifest.json` is not a study: skip it. `study_id` comes from the JSON, not the filename.
- Writer: `classification/export_phase8_batch.py` at commit `{full}`. It is not `export_studies_for_handoff.py`, which is the Phase-G driver.
- `pipeline_version` is `{pipeline}` in all {n} files.
- Models, identical in every file: `{mv_json}`
- Ensemble, as stored in every file: `{em_json}`
- The 20 Phase-G files carry `{PHASE_G_VERSION}` and come from a different run. Do not mix them with these.
- GT-swap study (the only one of the 22 known in this set): `{GT_SWAP[0]}`. Tag it wherever it is shown; keep it out of the demo set.
- The files are frozen: never edit or regenerate them.

**Attached** (as committed at `{a.commit}`; reference only, read them, do not run them)
- `export_phase8_batch.py`, `assemble_study_output.py`, `pdscd_output_schema.json`
- `SHA256SUMS`
{notes}
**Please reply with**
1. Received.
2. The result of `sha256sum -c SHA256SUMS` (PowerShell: `Get-FileHash -Algorithm SHA256 <file>` for each file).
3. Anything above that contradicts what you hold.

Then we hold the 20-minute J1 call to lock PLAN C section 2.
"""
    (out / "H1_message.md").write_text(msg, encoding="utf-8")
    shutil.make_archive(str(out), "zip", root_dir=str(out))

    print(f"Wrote {out}/ and {out}.zip")
    print(sums.rstrip())
    print("Next: send H1_message.md text + the files (or the zip). Done when the LLM owner confirms receipt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
