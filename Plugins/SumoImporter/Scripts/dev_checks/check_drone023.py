import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
truth_path = PROJECT_ROOT / "Plugins" / "SumoImporter" / "Scenarios" / "donghu_dense_uav_rain_fall" / "artifacts" / "episodes" / "episode_demo_dense_uav_rain_fall_90s" / "truth_frames.jsonl"
sample_ticks = {0, 100, 200, 300, 400, 500, 550, 600, 650, 700, 750, 800, 900}

with open(truth_path, "r", encoding="utf-8") as fh:
    for line in fh:
        frame = json.loads(line)
        tick = frame["tick"]
        if tick not in sample_ticks:
            continue
        for e in frame["entities"]:
            if e["entity_id"] != "drone_demo_a_023":
                continue
            tp = e.get("truth_pose") or {}
            pos = tp.get("position_enu_m", [0, 0, 0])
            ann = e.get("annotations") or {}
            sf = (ann.get("state_facets") or {}).get("activity") or {}
            activity = sf.get("activity_type", "")
            rp = e.get("render_presence") or {}
            submit = rp.get("submission_state", "")
            offstage = rp.get("offstage", False)
            speed = float(ann.get("speed_mps", 0.0) or 0.0)
            home = ((ann.get("state_facets") or {}).get("assignment") or {}).get("home_target")
            print(
                f"tick={tick:4d}  z={pos[2]:6.1f}  "
                f"activity={activity:20s}  speed={speed:.2f}  "
                f"xy=[{pos[0]:.1f}, {pos[1]:.1f}]  "
                f"submit={submit:20s}  offstage={offstage}  "
                f"home_z={home[2] if home else '?'}"
            )
