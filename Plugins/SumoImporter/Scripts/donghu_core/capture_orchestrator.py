from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .artifact_writer import write_json
from .interfaces import CaptureSidecar


def safe_frame_id(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_")


class CaptureOrchestrator:
    @staticmethod
    def frame_stem(frame: dict[str, Any]) -> str:
        return f"tick_{int(frame['tick']):06d}"

    @staticmethod
    def modality_output_dir(output_dir: Path, batch_id: str, camera_id: str, modality_id: str) -> Path:
        path = output_dir / batch_id / camera_id / modality_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def write_sidecar(path: Path, sidecar: CaptureSidecar | dict[str, Any]) -> None:
        payload = sidecar.to_dict() if isinstance(sidecar, CaptureSidecar) else dict(sidecar)
        write_json(path, payload)

    @staticmethod
    def build_sidecar(
        *,
        tick: int,
        frame_id: str,
        camera_id: str,
        camera_pose_enu_m: Sequence[float],
        weather: dict[str, Any],
        uav_runtime: dict[str, Any],
        entity_records: list[dict[str, Any]],
        capture_backend: str,
        output_path: str,
        extra: dict[str, Any] | None = None,
    ) -> CaptureSidecar:
        return CaptureSidecar(
            tick=int(tick),
            frame_id=str(frame_id),
            camera_id=str(camera_id),
            camera_pose_enu_m=[float(value) for value in camera_pose_enu_m[:3]],
            weather=dict(weather),
            uav_runtime=dict(uav_runtime),
            entity_records=list(entity_records),
            capture_backend=str(capture_backend),
            output_path=str(output_path),
            extra=dict(extra or {}),
        )
