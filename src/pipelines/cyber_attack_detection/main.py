import importlib
from datetime import datetime, timezone

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import MainWf
from core.logger import get_logger

log = get_logger(__name__)

SUB_WORKFLOWS = [
    "download",
    "preprocessing",
    "models",
    "training",
    "inference",
]


class CyberAttackDetection(MainWf):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        steps = self._resolve_steps(req.start_from)

        for step_name in steps:
            module_path = f"pipelines.cyber_attack_detection.{step_name}.facade"
            log.debug("ENTER sub-workflow: %s (%s)", step_name, module_path)
            started = datetime.now(timezone.utc)

            try:
                module = importlib.import_module(module_path)
                facade_cls = getattr(module, "Facade")
                resp = facade_cls().execute(req, resp)
            except Exception as exc:
                ended = datetime.now(timezone.utc)
                log.debug("EXCEPTION in sub-workflow %s: %s", step_name, exc)
                resp.tasks_executed.append({
                    "task_name": step_name,
                    "module_name": module_path,
                    "started": started.isoformat(),
                    "completed": ended.isoformat(),
                    "exception": str(exc),
                    "messages": {},
                })
                resp.success = False
                resp.message = f"Sub-workflow '{step_name}' raised: {exc}"
                return resp

            ended = datetime.now(timezone.utc)
            log.debug("EXIT  sub-workflow: %s — success=%s", step_name, resp.success)
            resp.tasks_executed.append({
                "task_name": step_name,
                "module_name": module_path,
                "started": started.isoformat(),
                "completed": ended.isoformat(),
                "exception": None,
                "messages": {"resp_message": resp.message},
            })

            if not resp.success:
                return resp

        resp.success = True
        resp.message = "Pipeline completed"
        return resp

    def _resolve_steps(self, start_from: str | None) -> list[str]:
        if start_from is None:
            return list(SUB_WORKFLOWS)
        if start_from not in SUB_WORKFLOWS:
            raise ValueError(
                f"Unknown sub-workflow '{start_from}'. "
                f"Available: {SUB_WORKFLOWS}"
            )
        idx = SUB_WORKFLOWS.index(start_from)
        return SUB_WORKFLOWS[idx:]


Pipeline = CyberAttackDetection
