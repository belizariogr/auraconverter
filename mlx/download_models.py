#!/usr/bin/env python3
"""Download Qwen3 TTS MLX models with live JSON progress (per-file + ETA)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


BASE_MODEL = "Qwen3-TTS-12Hz-1.7B-Base-8bit"
CUSTOM_MODEL = "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

MODELS = [
    {
        "id": "base",
        "label": "Base (ICL / voice cloning)",
        "repo": f"mlx-community/{BASE_MODEL}",
        "folder": BASE_MODEL,
        "required": True,
    },
    {
        "id": "custom",
        "label": "CustomVoice (prévias / speakers)",
        "repo": f"mlx-community/{CUSTOM_MODEL}",
        "folder": CUSTOM_MODEL,
        "required": True,
    },
]


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def model_present(models_dir: str, folder: str) -> bool:
    path = os.path.join(models_dir, folder)
    if not os.path.isdir(path):
        return False
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".safetensors", ".npz", ".json", ".model")):
                return True
    return False


def list_model_files(repo_id: str):
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
    siblings = [
        s
        for s in (info.siblings or [])
        if s.rfilename
        and not s.rfilename.endswith(".gitattributes")
        and s.rfilename != ".gitattributes"
    ]
    return siblings


class ProgressTracker:
    def __init__(self, overall_total: int):
        self.overall_total = max(0, int(overall_total))
        self.overall_done = 0
        self.started_at = time.time()
        self._last_emit = 0.0
        self.model_id = ""
        self.model_label = ""
        self.model_total = 0
        self.model_done_before_file = 0
        self.file_name = ""
        self.file_total = 0
        self.file_done = 0

    def begin_model(self, spec: dict, model_total: int, file_count: int) -> None:
        self.model_id = spec["id"]
        self.model_label = spec["label"]
        self.model_total = max(0, int(model_total))
        self.model_done_before_file = 0
        emit(
            {
                "type": "model_start",
                "model": spec["id"],
                "label": spec["label"],
                "repo": spec["repo"],
                "folder": spec["folder"],
                "totalBytes": self.model_total,
                "files": file_count,
                "overallDownloadedBytes": self.overall_done,
                "overallTotalBytes": self.overall_total,
                "percent": self._overall_percent(),
                "speedBytesPerSec": 0,
                "etaSeconds": None,
            }
        )

    def begin_file(self, filename: str, size: int) -> None:
        self.file_name = filename
        self.file_total = max(0, int(size))
        self.file_done = 0
        self._emit_progress(force=True, phase="Baixando arquivo")

    def add_file_bytes(self, n: int) -> None:
        if n <= 0:
            return
        self.file_done += n
        self.overall_done += n
        self._emit_progress(force=False, phase="Baixando arquivo")

    def end_file(self) -> None:
        # Align totals if server omitted Content-Length / metadata size.
        remaining = max(0, self.file_total - self.file_done)
        if remaining:
            self.file_done += remaining
            self.overall_done += remaining
        self.model_done_before_file += self.file_total or self.file_done
        self._emit_progress(force=True, phase="Arquivo concluído", event_type="file_done")

    def end_model(self, spec: dict) -> None:
        emit(
            {
                "type": "model_done",
                "model": spec["id"],
                "label": spec["label"],
                "folder": spec["folder"],
                "downloadedBytes": self.model_done_before_file,
                "totalBytes": self.model_total,
                "overallDownloadedBytes": self.overall_done,
                "overallTotalBytes": self.overall_total,
                "percent": self._overall_percent(),
                "speedBytesPerSec": self._speed(),
                "etaSeconds": self._eta(),
            }
        )

    def _overall_percent(self) -> float:
        if self.overall_total <= 0:
            return 0.0
        return round(100.0 * min(self.overall_done, self.overall_total) / self.overall_total, 2)

    def _model_percent(self) -> float:
        model_done = self.model_done_before_file + self.file_done
        if self.model_total <= 0:
            return 0.0
        return round(100.0 * min(model_done, self.model_total) / self.model_total, 2)

    def _file_percent(self) -> float:
        if self.file_total <= 0:
            return 0.0
        return round(100.0 * min(self.file_done, self.file_total) / self.file_total, 2)

    def _speed(self) -> float:
        elapsed = max(0.001, time.time() - self.started_at)
        return self.overall_done / elapsed

    def _eta(self) -> float | None:
        speed = self._speed()
        if speed <= 1 or self.overall_total <= 0:
            return None
        remaining = max(0, self.overall_total - self.overall_done)
        return round(remaining / speed, 1)

    def _emit_progress(self, *, force: bool, phase: str, event_type: str = "progress") -> None:
        now = time.time()
        if not force and (now - self._last_emit) < 0.2:
            return
        self._last_emit = now
        model_done = self.model_done_before_file + self.file_done
        emit(
            {
                "type": event_type,
                "model": self.model_id,
                "label": self.model_label,
                "file": self.file_name,
                "phase": phase,
                "fileDownloadedBytes": self.file_done,
                "fileTotalBytes": self.file_total,
                "filePercent": self._file_percent(),
                "downloadedBytes": model_done,
                "totalBytes": self.model_total,
                "modelPercent": self._model_percent(),
                "overallDownloadedBytes": self.overall_done,
                "overallTotalBytes": self.overall_total,
                "percent": self._overall_percent(),
                "speedBytesPerSec": round(self._speed(), 1),
                "etaSeconds": self._eta(),
            }
        )


def download_file(repo_id: str, filename: str, dest_dir: str, tracker: ProgressTracker) -> None:
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session

    dest_path = Path(dest_dir) / filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="model")
    headers = build_hf_headers()
    session = get_session()

    # Resume partial downloads when possible.
    resume_from = 0
    if tmp_path.exists():
        resume_from = tmp_path.stat().st_size
        if resume_from > 0:
            headers = {**headers, "Range": f"bytes={resume_from}-"}

    with session.get(url, headers=headers, stream=True, timeout=120) as response:
        # If resume was rejected, restart cleanly.
        if resume_from and response.status_code == 200:
            resume_from = 0
            try:
                tmp_path.unlink(missing_ok=True)
            except TypeError:
                if tmp_path.exists():
                    tmp_path.unlink()
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        remaining = int(content_length) if content_length else max(0, tracker.file_total - resume_from)
        expected_total = resume_from + remaining if remaining else tracker.file_total
        if expected_total > 0:
            tracker.file_total = expected_total

        if resume_from:
            tracker.file_done = resume_from
            tracker.overall_done += resume_from
            tracker._emit_progress(force=True, phase="Retomando arquivo")

        mode = "ab" if resume_from else "wb"
        with open(tmp_path, mode) as out:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                out.write(chunk)
                tracker.add_file_bytes(len(chunk))

    os.replace(tmp_path, dest_path)


def download_one(models_dir: str, spec: dict, tracker: ProgressTracker) -> None:
    local_dir = os.path.join(models_dir, spec["folder"])
    os.makedirs(local_dir, exist_ok=True)

    siblings = list_model_files(spec["repo"])
    model_total = sum(int(getattr(s, "size", 0) or 0) for s in siblings)
    tracker.begin_model(spec, model_total, len(siblings))

    for sibling in siblings:
        filename = sibling.rfilename
        size = int(getattr(sibling, "size", 0) or 0)
        tracker.begin_file(filename, size)
        download_file(spec["repo"], filename, local_dir, tracker)
        tracker.end_file()

    tracker.end_model(spec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    parser.add_argument(
        "--only",
        choices=["base", "custom", "all"],
        default="all",
        help="Which models to download",
    )
    args = parser.parse_args()

    models_dir = os.path.abspath(args.models_dir)
    os.makedirs(models_dir, exist_ok=True)

    selected = MODELS
    if args.only != "all":
        selected = [m for m in MODELS if m["id"] == args.only]

    emit({"type": "start", "modelsDir": models_dir, "count": len(selected)})

    try:
        # Probe totals for accurate overall ETA before downloading.
        pending = []
        overall_total = 0
        for spec in selected:
            if model_present(models_dir, spec["folder"]):
                emit(
                    {
                        "type": "model_skip",
                        "model": spec["id"],
                        "label": spec["label"],
                        "folder": spec["folder"],
                        "reason": "already_present",
                    }
                )
                continue
            siblings = list_model_files(spec["repo"])
            total = sum(int(getattr(s, "size", 0) or 0) for s in siblings)
            pending.append((spec, siblings, total))
            overall_total += total

        emit(
            {
                "type": "plan",
                "models": len(pending),
                "overallTotalBytes": overall_total,
            }
        )

        tracker = ProgressTracker(overall_total)
        for spec, _siblings, _total in pending:
            download_one(models_dir, spec, tracker)

        emit(
            {
                "type": "done",
                "modelsDir": models_dir,
                "ready": True,
                "overallDownloadedBytes": tracker.overall_done,
                "overallTotalBytes": tracker.overall_total,
                "percent": 100.0,
                "etaSeconds": 0,
            }
        )
        return 0
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
