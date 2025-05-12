from abc import ABC



class IdentificationModule(ABC):
    def __init__(
        self,
        debug: bool = False,
        debug_folder: str = ".tmp/debug",
    ):
        self.debug = debug
        self.debug_folder = debug_folder
        super().__init__()

