import importlib
from datetime import datetime, timezone

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import SubWf
from core.logger import get_logger

log = get_logger(__name__)

TASKS: list[str] = [
    "data_analysis",
    "cleaning",
    # "feature_engineering",
    # "encoding",
    # "scaling",
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

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        for task_name in TASKS:
            module_path = f"{__package__}.tasks.{task_name}"
            log.debug("ENTER task: %s (%s)", task_name, module_path)
            started = datetime.now(timezone.utc)

            try:
                resp = self._load_task(task_name).execute(req, resp)
            except Exception as exc:
                ended = datetime.now(timezone.utc)
                log.debug("EXCEPTION in task %s: %s", task_name, exc)
                resp.tasks_executed.append({
                    "task_name": task_name,
                    "module_name": module_path,
                    "started": started.isoformat(),
                    "completed": ended.isoformat(),
                    "exception": str(exc),
                    "messages": {},
                })
                resp.success = False
                resp.message = f"Task '{task_name}' raised: {exc}"
                return resp

            ended = datetime.now(timezone.utc)
            log.debug("EXIT  task: %s — success=%s", task_name, resp.success)
            resp.tasks_executed.append({
                "task_name": task_name,
                "module_name": module_path,
                "started": started.isoformat(),
                "completed": ended.isoformat(),
                "exception": None,
                "messages": {"resp_message": resp.message},
            })

            if not resp.success:
                return resp

        resp.message = "preprocessing completed"
        return resp


Facade = PreprocessingFacade
