import importlib

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import MainWf

SUB_WORKFLOWS = [
    "download",
    "preprocessing",
    "models",
    "training",
    "inference",
]


class CyberAttackDetection(MainWf):

    def execute(self, req: WfReq) -> WfResp:
        steps = self._resolve_steps(req.start_from)

        for step_name in steps:
            module = importlib.import_module(
                f"pipelines.cyber_attack_detection.{step_name}.facade"
            )
            facade_cls = getattr(module, "Facade")
            resp = facade_cls().execute(req)

            if not resp.success:
                return resp

            req.ctx_data.update(resp.ctx_data)

        return WfResp(success=True, message="Pipeline completed", ctx_data=req.ctx_data)

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
