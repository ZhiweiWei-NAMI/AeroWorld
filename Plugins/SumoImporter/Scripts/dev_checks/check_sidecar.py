import json, pathlib, math, sys

base = pathlib.Path(r"Saved/AirSim/episode_render_host/episode_demo_dense_uav_rain_fall_90s_low_fall_segment_15m_v1"
                    r"/site.intersection_a/demo_low_ground_8m/rgb")
for tick in [550, 600, 650, 750]:
    pat = list(base.glob(f"tick_{tick:06d}__*.json"))
    if not pat:
        continue
    d = json.loads(pat[0].read_text())
    cam = d["camera_pose_enu_m"]
    print(f"\n=== tick {tick}  camera={cam} ===")
    for r in d["entity_records"]:
        pos = r["position_enu_m"]
        z = pos[2]
        dx = pos[0] - cam[0]
        dy = pos[1] - cam[1]
        hdist = math.hypot(dx, dy)
        eid = r["entity_id"]
        show = False
        if "pedestrian_a_005" in eid:
            show = True
        if "drone" in eid and z < 25:
            show = True
        if show:
            print(f"  {eid}  mode={r['mode']}  z={z:.1f}  hdist={hdist:.1f}m  activity={r['activity_type']}")
    # also list any drone with hdist < 30 regardless of altitude
    for r in d["entity_records"]:
        pos = r["position_enu_m"]
        dx = pos[0] - cam[0]
        dy = pos[1] - cam[1]
        hdist = math.hypot(dx, dy)
        if "drone" in r["entity_id"] and hdist < 25:
            print(f"  NEAR: {r['entity_id']}  z={pos[2]:.1f}  hdist={hdist:.1f}m  activity={r['activity_type']}")
print()
