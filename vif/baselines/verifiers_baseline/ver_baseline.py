
from abc import abstractmethod

from vif.baselines.models import VerEvaluation


class TexVerBaseline:
    def __init__(
        self,
        *args, **kwargs
    ):
        super().__init__()

    @abstractmethod
    def get_feedback(self,ver_eval_input:VerEvaluation) -> VerEvaluation:
        pass
