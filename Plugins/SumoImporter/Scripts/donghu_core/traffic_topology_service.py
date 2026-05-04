from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LaneSamplePoint:
    s_m: float
    position_enu_m: tuple[float, float, float]
    yaw_deg: float
    lane_index: int


@dataclass(frozen=True)
class BundlePathSet:
    bundle_dir: Path
    lane_center_samples_path: Path
    lane_meta_path: Path
    lane_connections_path: Path

    @classmethod
    def from_bundle_dir(cls, bundle_dir: Path) -> "BundlePathSet":
        return cls(
            bundle_dir=bundle_dir,
            lane_center_samples_path=bundle_dir / "lane_center_samples.csv",
            lane_meta_path=bundle_dir / "lane_meta.csv",
            lane_connections_path=bundle_dir / "lane_connections.csv",
        )


def load_lane_samples_by_route(bundle_paths: BundlePathSet) -> dict[tuple[str, str], list[LaneSamplePoint]]:
    samples_by_route: dict[tuple[str, str], list[LaneSamplePoint]] = {}
    with bundle_paths.lane_center_samples_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            edge_id = str(row.get("edge_id") or "").strip()
            lane_id = str(row.get("lane_id") or "").strip()
            if not edge_id or not lane_id:
                continue
            samples_by_route.setdefault((edge_id, lane_id), []).append(
                LaneSamplePoint(
                    s_m=float(row.get("s_m") or 0.0),
                    position_enu_m=(
                        float(row.get("x_m") or 0.0),
                        float(row.get("y_m") or 0.0),
                        float(row.get("z_m") or 0.0),
                    ),
                    yaw_deg=float(row.get("yaw_deg") or 0.0),
                    lane_index=int(row.get("lane_index") or 0),
                )
            )
    for key in list(samples_by_route):
        samples_by_route[key] = sorted(samples_by_route[key], key=lambda item: item.s_m)
    return samples_by_route


@dataclass
class TrafficTopologyService:
    bundle_paths: BundlePathSet

    @classmethod
    def from_bundle_dir(cls, bundle_dir: Path) -> "TrafficTopologyService":
        return cls(bundle_paths=BundlePathSet.from_bundle_dir(bundle_dir))

    def load_lane_samples_by_route(self) -> dict[tuple[str, str], list[LaneSamplePoint]]:
        return load_lane_samples_by_route(self.bundle_paths)
