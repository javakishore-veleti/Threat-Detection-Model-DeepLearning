from dataclasses import dataclass, field

DEFAULT_DF_KEYS = ("train_df", "val_df", "test_df")
DEFAULT_LABEL_KEYS = ("train_labels", "val_labels", "test_labels")


@dataclass
class WfReq:
    pipeline: str = ""
    start_from: str | None = None
    resume: str | None = None
    config_path: str | None = None
    config: dict = field(default_factory=dict)
    ctx_data: dict = field(default_factory=dict)

    @property
    def df_keys(self) -> tuple[str, ...]:
        """Derive DF keys from config splits, falling back to defaults."""
        if not self.config:
            return DEFAULT_DF_KEYS
        from core.config import split_keys
        dk, _ = split_keys(self.config)
        return dk

    @property
    def label_keys(self) -> tuple[str, ...]:
        """Derive label keys from config splits, falling back to defaults."""
        if not self.config:
            return DEFAULT_LABEL_KEYS
        from core.config import split_keys
        _, lk = split_keys(self.config)
        return lk


@dataclass
class WfResp:
    success: bool = True
    message: str = ""
    ctx_data: dict = field(default_factory=dict)
    tasks_executed: list = field(default_factory=list)
