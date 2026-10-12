"""Run all L1 v3 arms."""
import concurrent.futures as futures, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python"
def main():
    scripts = sorted((ROOT/"scripts/arms").glob("L1-*/seed*/grp_*/arm_*.py"))
    print(json.dumps({"arms": len(scripts)}), flush=True)
    results = []
    def run_one(s):
        rel = s.resolve().relative_to((ROOT/"scripts/arms").resolve()).with_suffix("")
        manifest = ROOT/"arms"/rel/"manifest.json"
        if manifest.is_file():
            return {"script": str(s.relative_to(ROOT)), "rc": 0, "skipped": True}
        t0 = time.time()
        p = subprocess.run([PYTHON, "-B", str(s)], cwd=str(ROOT), capture_output=True, text=True)
        return {"script": str(s.relative_to(ROOT)), "rc": p.returncode,
                "skipped": False, "s": round(time.time()-t0, 1),
                "tail": (p.stderr or p.stdout).strip().splitlines()[-1][:200] if (p.stderr or p.stdout).strip() else ""}
    with futures.ThreadPoolExecutor(max_workers=24) as pool:
        for r in pool.map(run_one, scripts):
            results.append(r)
            if r["rc"] != 0 and not r.get("skipped"):
                print(f"  FAIL {r['script']}: {r.get('tail','')[:150]}", flush=True)
    bad = [r for r in results if r["rc"] != 0 and not r.get("skipped")]
    print(json.dumps({"total": len(results), "failed": len(bad),
                       "skipped": sum(1 for r in results if r.get("skipped"))}))
    return 1 if bad else 0
if __name__ == "__main__":
    raise SystemExit(main())
