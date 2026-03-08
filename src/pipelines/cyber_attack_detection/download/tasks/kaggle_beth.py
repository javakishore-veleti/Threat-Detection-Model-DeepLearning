"""Download the BETH dataset from Kaggle into ~/python_venvs/datasets/kaggle/beth-dataset/."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.logger import get_logger

log = get_logger(__name__)

KAGGLE_DATASET = "katehighnam/beth-dataset"
SOURCE_URL = "https://www.kaggle.com/datasets/katehighnam/beth-dataset"
DEST_DIR = Path.home() / "python_venvs" / "datasets" / "kaggle" / "beth-dataset"
MARKER = "DOWNLOAD_COMPLETED.json"


class KaggleBeth(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        marker_path = DEST_DIR / MARKER

        if marker_path.exists():
            meta = json.loads(marker_path.read_text())
            log.debug("Skipping download — marker exists (downloaded %s)", meta.get("download_ended"))
            resp.message = f"Already downloaded at {meta.get('download_ended')}"
            resp.ctx_data["raw_data_path"] = str(DEST_DIR)
            return resp

        self._validate_credentials()
        DEST_DIR.mkdir(parents=True, exist_ok=True)

        log.debug("Downloading %s -> %s", KAGGLE_DATASET, DEST_DIR)
        start = datetime.now(timezone.utc)
        self._download()
        end = datetime.now(timezone.utc)

        file_count = sum(1 for f in DEST_DIR.iterdir() if f.is_file() and f.name != MARKER)

        meta = {
            "source_url": SOURCE_URL,
            "kaggle_dataset": KAGGLE_DATASET,
            "download_started": start.isoformat(),
            "download_ended": end.isoformat(),
            "file_count": file_count,
        }
        marker_path.write_text(json.dumps(meta, indent=2))

        log.debug("Download complete — %d file(s) in %.1fs", file_count, (end - start).total_seconds())
        resp.message = f"Downloaded {file_count} file(s) in {(end - start).total_seconds():.1f}s"
        resp.ctx_data["raw_data_path"] = str(DEST_DIR)
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
    def _download():
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=str(DEST_DIR), unzip=True)


Task = KaggleBeth
