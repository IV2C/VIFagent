
from abc import abstractmethod

from vif.baselines.models import VerEvaluation


class TexVerBaseline:
    def __init__(
        self,
        *args, **kwargs
    ):
        super().__init__()

    @abstractmethod
    def assess_customization(self,ver_eval_input:VerEvaluation) -> VerEvaluation:
        pass
    
    @abstractmethod
    def get_config_metadata(self)->dict:
        pass
