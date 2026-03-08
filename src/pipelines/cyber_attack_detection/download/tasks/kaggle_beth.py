"""Download a dataset from Kaggle. All paths and identifiers come from req.config."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.config import get_cfg
from core.logger import get_logger

log = get_logger(__name__)

MARKER = "DOWNLOAD_COMPLETED.json"


class KaggleBeth(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        cfg = req.config
        kaggle_dataset = get_cfg(cfg, "dataset.source.dataset_id", "")
        source_url = get_cfg(cfg, "dataset.source.url", "")
        dest_dir = Path(get_cfg(cfg, "dataset.paths.data_dir", ""))

        if not kaggle_dataset or not dest_dir.name:
            resp.success = False
            resp.message = "dataset.source.dataset_id and dataset.paths.data_dir required in config"
            return resp

        marker_path = dest_dir / MARKER

        if marker_path.exists():
            meta = json.loads(marker_path.read_text())
            log.debug("Skipping download — marker exists (downloaded %s)", meta.get("download_ended"))
            resp.message = f"Already downloaded at {meta.get('download_ended')}"
            resp.ctx_data["raw_data_path"] = str(dest_dir)
            return resp

        self._validate_credentials()
        dest_dir.mkdir(parents=True, exist_ok=True)

        log.debug("Downloading %s -> %s", kaggle_dataset, dest_dir)
        start = datetime.now(timezone.utc)
        self._download(kaggle_dataset, dest_dir)
        end = datetime.now(timezone.utc)

        file_count = sum(1 for f in dest_dir.iterdir() if f.is_file() and f.name != MARKER)

        meta = {
            "source_url": source_url,
            "kaggle_dataset": kaggle_dataset,
            "download_started": start.isoformat(),
            "download_ended": end.isoformat(),
            "file_count": file_count,
        }
        marker_path.write_text(json.dumps(meta, indent=2))

        log.debug("Download complete — %d file(s) in %.1fs", file_count, (end - start).total_seconds())
        resp.message = f"Downloaded {file_count} file(s) in {(end - start).total_seconds():.1f}s"
        resp.ctx_data["raw_data_path"] = str(dest_dir)
        return resp

    @staticmethod
    def _validate_credentials():
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
        if not has_env and not kaggle_json.exists():
            raise EnvironmentError(
                "Kaggle credentials not found. Either set KAGGLE_USERNAME and "
                "KAGGLE_KEY env vars, or place a kaggle.json in ~/.kaggle/"
            )

    @staticmethod
    def _download(dataset_id: str, dest: Path):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_id, path=str(dest), unzip=True)


Task = KaggleBeth
