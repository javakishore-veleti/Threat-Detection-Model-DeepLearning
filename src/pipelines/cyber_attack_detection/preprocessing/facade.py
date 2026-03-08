import importlib

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import SubWf

TASKS: list[str] = [
    "cleaning",
    "feature_engineering",
    "encoding",
    "scaling",
]


class PreprocessingFacade(SubWf):

    def __init__(self):
        self._task_cache: dict = {}

    def _load_task(self, task_name: str):
        if task_name not in self._task_cache:
            module = importlib.import_module(
                f".tasks.{task_name}", package=__package__
            )
            self._task_cache[task_name] = getattr(module, "Task")()
        return self._task_cache[task_name]

    def execute(self, req: WfReq) -> WfResp:
        for task_name in TASKS:
            resp = self._load_task(task_name).execute(req)
            if not resp.success:
                return resp
            req.ctx_data.update(resp.ctx_data)
        return WfResp(success=True, message="preprocessing completed", ctx_data=req.ctx_data)


Facade = PreprocessingFacade
