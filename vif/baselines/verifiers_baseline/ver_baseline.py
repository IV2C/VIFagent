
from abc import abstractmethod

from vif.baselines.models import VerEvaluationOutput


class TexVerBaseline:
    def __init__(
        self,
        *args, **kwargs
    ):
        super().__init__()

    @abstractmethod
    def get_feedback(self, code: str, instruction: str) -> VerEvaluationOutput:
        pass
