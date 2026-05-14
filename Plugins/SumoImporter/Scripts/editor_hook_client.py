#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
REMOTE_EXEC_RELATIVE_PATH = Path("Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py")
DEFAULT_DISCOVERY_TIMEOUT_S = 15.0
DEFAULT_CAPTURE_TIMEOUT_S = 15.0
DEFAULT_COMMAND_PORT_CANDIDATES = (6976, 7076, 7176, 7276, 7376, 7476, 7576, 7676)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_path(path_value: Any) -> str:
    try:
        return str(Path(str(path_value)).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path_value).replace("\\", "/").lower()


def _rotation_deg_payload(rotation_deg: dict[str, float]) -> dict[str, float]:
    return {
        "roll_deg": float(rotation_deg.get("roll_deg", rotation_deg.get("roll", 0.0))),
        "pitch_deg": float(rotation_deg.get("pitch_deg", rotation_deg.get("pitch", 0.0))),
        "yaw_deg": float(rotation_deg.get("yaw_deg", rotation_deg.get("yaw", 0.0))),
    }


def _project_name(project_root: Path) -> str:
    for uproject_path in sorted(project_root.glob("*.uproject")):
        return uproject_path.stem
    return project_root.name


def _engine_association_names(project_root: Path) -> list[str]:
    for uproject_path in sorted(project_root.glob("*.uproject")):
        doc = _load_json(uproject_path)
        raw = str(doc.get("EngineAssociation") or "").strip()
        if not raw:
            return []
        names = [raw]
        if not raw.upper().startswith("UE_"):
            names.append(f"UE_{raw}")
        return names
    return []


def _remote_exec_script_path(engine_root: Path) -> Path:
    return engine_root / REMOTE_EXEC_RELATIVE_PATH


def _resolve_engine_root(project_root: Path) -> Path:
    env_value = os.environ.get("UE_ENGINE_ROOT", "").strip()
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.name.lower() != "engine" and (candidate / "Engine").exists():
            candidate = candidate / "Engine"
        if _remote_exec_script_path(candidate).exists():
            return candidate

    association_names = _engine_association_names(project_root)
    launcher_installed_path = (
        Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
        / "Epic"
        / "UnrealEngineLauncher"
        / "LauncherInstalled.dat"
    )
    if association_names and launcher_installed_path.exists():
        try:
            launcher_doc = _load_json(launcher_installed_path)
        except Exception:
            launcher_doc = {}
        for installation in launcher_doc.get("InstallationList", []):
            app_name = str(installation.get("AppName") or "").strip()
            artifact_id = str(installation.get("ArtifactId") or "").strip()
            if app_name not in association_names and artifact_id not in association_names:
                continue
            install_location = Path(str(installation.get("InstallLocation") or "")).expanduser()
            candidate = install_location / "Engine" if (install_location / "Engine").exists() else install_location
            if _remote_exec_script_path(candidate).exists():
                return candidate

    candidates: list[Path] = []
    for association_name in association_names:
        candidates.extend(
            [
                Path(f"E:/{association_name}/Engine"),
                Path(f"D:/{association_name}/Engine"),
                Path(f"C:/Program Files/Epic Games/{association_name}/Engine"),
                Path(f"E:/Epic Games/{association_name}/Engine"),
                Path(f"D:/Epic Games/{association_name}/Engine"),
            ]
        )
    for candidate in candidates:
        if _remote_exec_script_path(candidate).exists():
            return candidate

    association_text = ", ".join(association_names) if association_names else "<unknown>"
    raise RuntimeError(
        f"Unable to resolve Unreal Engine root for '{project_root}'. "
        f"Tried EngineAssociation values: {association_text}. "
        "Set UE_ENGINE_ROOT or update LauncherInstalled.dat."
    )


def _import_remote_execution(project_root: Path) -> tuple[Path, Any]:
    engine_root = _resolve_engine_root(project_root)
    remote_exec_dir = _remote_exec_script_path(engine_root).parent
    if str(remote_exec_dir) not in sys.path:
        sys.path.insert(0, str(remote_exec_dir))
    import remote_execution  # type: ignore

    return engine_root, remote_execution


def _can_bind_tcp_endpoint(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, int(port)))
        return True
    except OSError:
        return False


def _select_command_endpoint(remote_execution: Any) -> tuple[str, int]:
    default_host, default_port = remote_execution.DEFAULT_COMMAND_ENDPOINT
    host = str(os.environ.get("UE_PYTHON_REMOTE_COMMAND_HOST") or default_host or "127.0.0.1")
    requested_port = str(os.environ.get("UE_PYTHON_REMOTE_COMMAND_PORT") or "").strip()
    if requested_port:
        port = int(requested_port)
        if not _can_bind_tcp_endpoint(host, port):
            raise RuntimeError(f"UE Python remote command endpoint is not bindable: {host}:{port}")
        return host, port

    candidates = [int(default_port), *DEFAULT_COMMAND_PORT_CANDIDATES]
    for port in candidates:
        if _can_bind_tcp_endpoint(host, port):
            return host, int(port)
    raise RuntimeError(f"No bindable UE Python remote command endpoint found for host {host}: {candidates}")


class UnrealEditorRemoteExecution:
    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        discovery_timeout_s: float = DEFAULT_DISCOVERY_TIMEOUT_S,
    ) -> None:
        self.project_root = project_root.resolve()
        self.discovery_timeout_s = max(1.0, float(discovery_timeout_s))
        self.engine_root, self.remote_execution = _import_remote_execution(self.project_root)
        self._session: Any | None = None
        self._remote_node: dict[str, Any] | None = None

    @staticmethod
    def _is_retryable_command_error(exc: Exception) -> bool:
        message = str(exc)
        if isinstance(exc, OSError):
            return True
        retry_markers = (
            "Remote party failed to attempt the command socket connection",
            "Remote party failed to send a valid response",
            "WinError 10057",
            "WinError 10054",
            "WinError 10053",
            "socket",
        )
        return any(marker in message for marker in retry_markers)

    def _matches_project(self, node: dict[str, Any]) -> bool:
        return _normalized_path(node.get("project_root") or "") == _normalized_path(self.project_root)

    def _pick_remote_node(self, nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
        matching = [node for node in nodes if self._matches_project(node)]
        if matching:
            return matching[0]
        if len(nodes) == 1:
            return nodes[0]
        return None

    def connect(self) -> None:
        if self._session is not None and self._session.has_command_connection():
            return

        self.close()
        config = self.remote_execution.RemoteExecutionConfig()
        config.command_endpoint = _select_command_endpoint(self.remote_execution)
        self._session = self.remote_execution.RemoteExecution(config)
        self._session.start()

        deadline = time.perf_counter() + self.discovery_timeout_s
        chosen_node: dict[str, Any] | None = None
        while time.perf_counter() < deadline:
            chosen_node = self._pick_remote_node(list(self._session.remote_nodes))
            if chosen_node is not None:
                break
            time.sleep(0.5)

        if chosen_node is None:
            self.close()
            raise RuntimeError(
                "Unable to discover a matching Unreal Editor remote execution node. "
                "Enable Python remote execution and reopen the editor."
            )

        self._session.open_command_connection(str(chosen_node["node_id"]))
        self._remote_node = dict(chosen_node)

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.stop()
            finally:
                self._session = None
                self._remote_node = None

    def run_python(
        self,
        command: str,
        *,
        unattended: bool = False,
        raise_on_failure: bool = True,
    ) -> dict[str, Any]:
        attempts = 2
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                self.connect()
                assert self._session is not None
                return self._session.run_command(command, unattended=unattended, raise_on_failure=raise_on_failure)
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= attempts or not self._is_retryable_command_error(exc):
                    raise
                self.close()
                time.sleep(0.5)
        assert last_exc is not None
        raise last_exc


class FixedWorldCaptureEditorHook:
    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        discovery_timeout_s: float = DEFAULT_DISCOVERY_TIMEOUT_S,
        capture_timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S,
    ) -> None:
        self.project_root = project_root.resolve()
        self.capture_timeout_s = max(1.0, float(capture_timeout_s))
        self.project_name = _project_name(self.project_root)
        self.log_path = self.project_root / "Saved" / "Logs" / f"{self.project_name}.log"
        self.remote = UnrealEditorRemoteExecution(
            project_root=self.project_root,
            discovery_timeout_s=discovery_timeout_s,
        )

    def close(self) -> None:
        self.remote.close()

    @staticmethod
    def _make_request_json(map_id: str, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "api_version": "1.0",
            "request_id": request_id,
            "map_id": map_id,
            "payload": payload,
        }

    def _run_console_json_command(self, command_name: str, request_json: dict[str, Any]) -> dict[str, Any]:
        request_json_text = json.dumps(request_json, separators=(",", ":"), ensure_ascii=True)
        python_command = f"""
import json
import unreal

worlds = unreal.EditorLevelLibrary.get_pie_worlds(False)
if not worlds:
    raise RuntimeError("No PIE world available for fixed world capture.")

world = worlds[0]
payload = json.loads({request_json_text!r})
unreal.SystemLibrary.execute_console_command(
    world,
    {command_name!r} + " " + json.dumps(payload, separators=(",", ":")),
    None,
)
print("EDITOR_HOOK_SENT", {command_name!r}, payload.get("request_id"))
"""
        return self.remote.run_python(python_command, unattended=False, raise_on_failure=True)

    def _run_python_json(self, python_command: str) -> Any:
        result = self.remote.run_python(python_command, unattended=False, raise_on_failure=True)
        outputs = list(result.get("output") or [])
        texts: list[str] = []
        for item in outputs:
            text = str((item or {}).get("output") or "").strip()
            if text:
                texts.append(text)
        for text in reversed(texts):
            try:
                return json.loads(text)
            except Exception:
                continue
        raise RuntimeError(f"Remote Python command did not emit JSON. Raw output: {texts!r}")

    def ensure_fixed_world_camera(
        self,
        *,
        map_id: str,
        asset_id: str,
        logical_asset_id: str,
        position_enu_m: list[float],
        rotation_deg: dict[str, float],
    ) -> dict[str, Any]:
        request_json = self._make_request_json(
            map_id,
            f"spawn_{asset_id}",
            {
                "asset_id": asset_id,
                "logical_asset_id": logical_asset_id,
                "pose_enu_m": {
                    "position_enu_m": [float(value) for value in position_enu_m],
                    "rotation_deg": _rotation_deg_payload(rotation_deg),
                },
            },
        )
        return self._run_console_json_command("aero.spawn_asset_json", request_json)

    def remove_asset(
        self,
        *,
        map_id: str,
        asset_id: str,
    ) -> dict[str, Any]:
        request_json = self._make_request_json(
            map_id,
            f"remove_{asset_id}",
            {
                "asset_id": asset_id,
            },
        )
        return self._run_console_json_command("aero.remove_asset_json", request_json)

    def capture_modality(
        self,
        *,
        map_id: str,
        asset_id: str,
        modality: str,
        output_path: Path,
        width: int,
        height: int,
        fov_degrees: float,
        semantic_rules_path: Path | str | None = None,
        semantic_audit_path: Path | str | None = None,
    ) -> dict[str, Any]:
        normalized_modality = str(modality or "rgb").strip().lower()
        if normalized_modality not in {"rgb", "depth", "seg"}:
            raise ValueError(f"Unsupported fixed world capture modality: {modality!r}")
        absolute_output_path = output_path.resolve()
        absolute_output_path.parent.mkdir(parents=True, exist_ok=True)
        previous_mtime = absolute_output_path.stat().st_mtime if absolute_output_path.exists() else None
        request_json = self._make_request_json(
            map_id,
            f"capture_{asset_id}_{normalized_modality}",
            {
                "asset_id": asset_id,
                "modality": normalized_modality,
                "output_path": str(absolute_output_path).replace("\\", "/"),
                "width": int(width),
                "height": int(height),
                "fov_degrees": float(fov_degrees),
            },
        )
        if semantic_rules_path is not None and str(semantic_rules_path).strip():
            request_json["payload"]["semantic_rules_path"] = str(Path(str(semantic_rules_path)).resolve()).replace("\\", "/")
        if semantic_audit_path is not None and str(semantic_audit_path).strip():
            request_json["payload"]["semantic_audit_path"] = str(Path(str(semantic_audit_path)).resolve()).replace("\\", "/")
        response = self._run_console_json_command("aero.capture_world_camera_json", request_json)
        self._wait_for_output(absolute_output_path, previous_mtime)
        return {
            "editor_hook_response": response,
            "modality": normalized_modality,
            "output_path": str(absolute_output_path),
            "width": int(width),
            "height": int(height),
            "fov_degrees": float(fov_degrees),
            "semantic_rules_path": str(Path(str(semantic_rules_path)).resolve()) if semantic_rules_path else "",
            "semantic_audit_path": str(Path(str(semantic_audit_path)).resolve()) if semantic_audit_path else "",
        }

    def capture_rgb(
        self,
        *,
        map_id: str,
        asset_id: str,
        output_path: Path,
        width: int,
        height: int,
        fov_degrees: float,
    ) -> dict[str, Any]:
        return self.capture_modality(
            map_id=map_id,
            asset_id=asset_id,
            modality="rgb",
            output_path=output_path,
            width=width,
            height=height,
            fov_degrees=fov_degrees,
        )

    def semantic_stencil_audit(
        self,
        *,
        map_id: str,
        semantic_rules_path: Path | str,
        semantic_audit_path: Path | str,
        assign: bool = False,
    ) -> dict[str, Any]:
        request_json = self._make_request_json(
            map_id,
            "semantic_stencil_audit",
            {
                "semantic_rules_path": str(Path(str(semantic_rules_path)).resolve()).replace("\\", "/"),
                "semantic_audit_path": str(Path(str(semantic_audit_path)).resolve()).replace("\\", "/"),
                "assign": bool(assign),
            },
        )
        response = self._run_console_json_command("aero.semantic_stencil_audit_json", request_json)
        self._wait_for_output(Path(str(semantic_audit_path)).resolve(), None)
        return {
            "editor_hook_response": response,
            "semantic_rules_path": str(Path(str(semantic_rules_path)).resolve()),
            "semantic_audit_path": str(Path(str(semantic_audit_path)).resolve()),
            "assign": bool(assign),
        }

    def _wait_for_output(self, output_path: Path, previous_mtime: float | None) -> None:
        deadline = time.perf_counter() + self.capture_timeout_s
        while time.perf_counter() < deadline:
            if output_path.exists():
                stat = output_path.stat()
                if stat.st_size > 0 and (previous_mtime is None or stat.st_mtime > previous_mtime):
                    return
            time.sleep(0.25)

        log_excerpt = ""
        if self.log_path.exists():
            try:
                lines = self.log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                excerpt_lines = [
                    line
                    for line in lines
                    if "simAeroCaptureWorldCamera" in line or "fixed_world_camera" in line or "CaptureWorldCamera" in line
                ]
                if excerpt_lines:
                    log_excerpt = "\n".join(excerpt_lines[-10:])
            except Exception:
                log_excerpt = ""

        message = f"Fixed world capture did not produce '{output_path}' within {self.capture_timeout_s:.1f}s."
        if log_excerpt:
            message += f"\nRelevant log lines:\n{log_excerpt}"
        raise RuntimeError(message)

    def apply_weather(
        self,
        *,
        map_id: str,
        payload: dict[str, Any],
        request_id: str = "apply_weather",
    ) -> dict[str, Any]:
        request_json = self._make_request_json(
            map_id,
            request_id,
            payload,
        )
        return self._run_console_json_command("aero.apply_weather_json", request_json)

    def inspect_capture_vehicle_actors(
        self,
        *,
        vehicle_names: list[str],
        world_origin_cm: list[float],
    ) -> dict[str, Any]:
        names_text = json.dumps([str(value) for value in vehicle_names], ensure_ascii=True)
        origin_text = json.dumps([float(value) for value in world_origin_cm], ensure_ascii=True)
        python_command = f"""
import json
import unreal

vehicle_names = json.loads({names_text!r})
world_origin_cm = json.loads({origin_text!r})

worlds = unreal.EditorLevelLibrary.get_pie_worlds(False)
if not worlds:
    raise RuntimeError("No PIE world available for AirSim capture vehicle actor inspection.")

world = worlds[0]
pawns = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Pawn)

def _safe_label(actor):
    try:
        return actor.get_actor_label()
    except Exception:
        return ""

def _serialize_pawn(actor):
    loc = actor.get_actor_location()
    rot = actor.get_actor_rotation()
    world_cm = [float(loc.x), float(loc.y), float(loc.z)]
    return {{
        "actor_name": actor.get_name(),
        "actor_label": _safe_label(actor),
        "class_name": actor.get_class().get_name(),
        "position_world_cm": world_cm,
        "position_enu_m": [
            (world_cm[0] - float(world_origin_cm[0])) / 100.0,
            (world_cm[1] - float(world_origin_cm[1])) / 100.0,
            (world_cm[2] - float(world_origin_cm[2])) / 100.0,
        ],
        "rotation_deg": {{
            "pitch_deg": float(rot.pitch),
            "yaw_deg": float(rot.yaw),
            "roll_deg": float(rot.roll),
        }},
    }}

all_flying_pawns = []
serialized_pawns = []
for pawn in pawns:
    row = _serialize_pawn(pawn)
    serialized_pawns.append(row)
    cls = str(row.get("class_name") or "")
    if "FlyingPawn" in cls or "ComputerVisionPawn" in cls:
        all_flying_pawns.append(row)

vehicles = {{}}
for vehicle_name in vehicle_names:
    lower_name = vehicle_name.lower()
    best = None
    fallback_names = []
    for row in serialized_pawns:
        actor_name = str(row.get("actor_name") or "")
        actor_label = str(row.get("actor_label") or "")
        hay = (actor_name + " " + actor_label).lower()
        if actor_name == vehicle_name or actor_label == vehicle_name:
            best = row
            break
        if actor_name.startswith(vehicle_name + "_") or actor_label.startswith(vehicle_name):
            if best is None:
                best = row
            if len(fallback_names) < 3:
                fallback_names.append({{
                    "actor_name": actor_name,
                    "actor_label": actor_label,
                }})
            continue
        if lower_name in hay and len(fallback_names) < 3:
            fallback_names.append({{
                "actor_name": actor_name,
                "actor_label": actor_label,
            }})
    vehicles[vehicle_name] = {{
        "found": best is not None,
        "actor": best,
        "name_candidates": fallback_names,
    }}

print(json.dumps({{
    "vehicle_count": len(vehicle_names),
    "pawn_count": len(serialized_pawns),
    "flying_pawns": all_flying_pawns[:50],
    "vehicles": vehicles,
}}, ensure_ascii=False))
"""
        return self._run_python_json(python_command)
