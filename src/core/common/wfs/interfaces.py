from abc import ABC, abstractmethod

from core.common.wfs.dtos import WfReq, WfResp


class MainWf(ABC):
    """Interface for top-level pipeline workflows.
    Implemented by each pipeline's main class (e.g., CyberAttackDetection).
    """

    @abstractmethod
    def execute(self, req: WfReq) -> WfResp: ...


class SubWf(ABC):
    """Interface for sub-workflow facades.
    Implemented by each sub-module's facade.py (e.g., PreprocessingFacade).
    """

    @abstractmethod
    def execute(self, req: WfReq) -> WfResp: ...


class WfTask(ABC):
    """Interface for individual tasks within a sub-workflow.
    Implemented by each task file (e.g., CleaningTask).
    """

    @abstractmethod
    def execute(self, req: WfReq) -> WfResp: ...
