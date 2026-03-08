"""Load pipeline YAML configuration and provide dot-path access helpers."""

from pathlib import Path
from typing import Any

import yaml

from core.logger import get_logger

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(pipeline: str, override_path: str | None = None) -> dict:
    """Load YAML config for a pipeline.

    Resolution order:
      1. override_path (if provided)
      2. configs/<pipeline>/default.yaml (relative to project root)
    """
    if override_path:
        config_path = Path(override_path).expanduser()
    else:
        config_path = _PROJECT_ROOT / "configs" / pipeline / "default.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    _expand_paths(cfg)
    log.debug("Loaded config from %s (%d top-level keys)", config_path, len(cfg))
    return cfg


def get_cfg(cfg: dict, dot_path: str, default: Any = None) -> Any:
    """Access nested config values via dot notation.

    >>> get_cfg(cfg, "columns.labels.target")  # returns "evil"
    >>> get_cfg(cfg, "missing.key", "fallback")  # returns "fallback"
    """
    keys = dot_path.split(".")
    node = cfg
    for key in keys:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


def split_keys(cfg: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive (df_keys, label_keys) from config splits + prefixes.

    Returns:
        (("train_df", "val_df", "test_df"),
         ("train_labels", "val_labels", "test_labels"))
    """
    splits = cfg.get("splits", {})
    prefixes = cfg.get("split_prefixes", {})

    default_prefix_map = {"training": "train", "validation": "val", "testing": "test"}

    df_keys = []
    label_keys = []
    for split_name in splits:
        prefix = prefixes.get(split_name, default_prefix_map.get(split_name, split_name))
        df_keys.append(f"{prefix}_df")
        label_keys.append(f"{prefix}_labels")

    return tuple(df_keys), tuple(label_keys)


def col_names(cfg: dict, category: str) -> list[str]:
    """Get column name list for a category from config.

    >>> col_names(cfg, "true_numeric")       # ["timestamp", "argsNum", "returnValue"]
    >>> col_names(cfg, "categorical_ids")    # ["processId", ...]
    >>> col_names(cfg, "complex_drop")       # ["stackAddresses"]
    """
    col_cfg = get_cfg(cfg, f"columns.{category}", {})
    if isinstance(col_cfg, dict):
        return list(col_cfg.get("names", []))
    return list(col_cfg)


def label_cols(cfg: dict) -> list[str]:
    """Return [target, auxiliary] label column names."""
    labels = get_cfg(cfg, "columns.labels", {})
    return [labels.get("target", ""), labels.get("auxiliary", "")]


def build_column_classification(cfg: dict) -> dict:
    """Build the full column_classification dict from config for reports/ctx_data."""
    columns_cfg = cfg.get("columns", {})
    classification = {}

    for cat_key in ("true_numeric", "categorical_id", "string_categorical",
                    "complex_drop", "parse_then_drop"):
        cfg_key = cat_key + "s" if cat_key == "categorical_id" else cat_key
        section = columns_cfg.get(cfg_key, {})
        classification[cat_key] = {
            "columns": list(section.get("names", [])),
            "treatment": section.get("treatment", ""),
            "detail": section.get("detail", ""),
        }

    labels_section = columns_cfg.get("labels", {})
    classification["labels"] = {
        "columns": [labels_section.get("target", ""), labels_section.get("auxiliary", "")],
        "treatment": labels_section.get("treatment", ""),
        "detail": labels_section.get("detail", ""),
    }

    return classification


def _expand_paths(obj: Any, _parent_key: str = "") -> None:
    """Recursively expand ~ in string values that look like paths."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, str) and ("~" in val or val.startswith("/")):
                if key.endswith(("_dir", "_path", "dir", "path")):
                    obj[key] = str(Path(val).expanduser())
            else:
                _expand_paths(val, key)
    elif isinstance(obj, list):
        for item in obj:
            _expand_paths(item, _parent_key)
