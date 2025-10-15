

from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from loguru import logger

class VerifierExecutor():
    def __init__(self, baseline:TexVerBaseline):
        from datasets import load_dataset


        ds = load_dataset("CharlyR/vtikz", "tikz", split="test")
        ds = ds.select_columns(["id","type","instruction","code","image_solution","image_input","code_solution"])

        
        pass