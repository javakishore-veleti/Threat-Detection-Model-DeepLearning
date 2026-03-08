from dataclasses import dataclass, field


@dataclass
class WfReq:
    pipeline: str = ""
    start_from: str | None = None
    resume: str | None = None
    config_path: str | None = None
    config: dict = field(default_factory=dict)
    ctx_data: dict = field(default_factory=dict)


@dataclass
class WfResp:
    success: bool = True
    message: str = ""
    ctx_data: dict = field(default_factory=dict)
    tasks_executed: list = field(default_factory=list)
