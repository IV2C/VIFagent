import os
import uuid



from PIL import Image

from loguru import logger

from vif.CodeMapper.feature import CodeImageMapping, MappedCode
from vif.CodeMapper.mutation.tex_mutant_creator import TexMappingMutantCreator, TexMutantCreator
from vif.utils.detection_utils import dsim_box


class MappingModule:
    def __init__(
        self,
        mutant_creator: TexMutantCreator = TexMappingMutantCreator(),
        debug: bool = False,
        debug_folder: str = ".tmp/debug",
    ):
        self.mutant_creator = mutant_creator
        self.debug = debug
        self.debug_folder = debug_folder
        super().__init__()

    def identify(
        self, code: str, features: list[str], base_image: Image.Image
    ) -> MappedCode:
        pass

    def __str__(self):
        return (
            f"{self.__class__.__name__}(mutant_creator={self.mutant_creator.__class__.__name__}, "
            f"debug={self.debug}, debug_folder='{self.debug_folder}')"
        )


class LLMMappingModule(MappingModule):

    def __init__(
        self,
        *,
        model,
        client,
        temperature=0.3,
        mutant_creator: TexMutantCreator = TexMappingMutantCreator(),
        debug=False,
        debug_folder=".tmp/debug",
    ):
        self.identification_client = client
        self.identification_model = model
        self.identification_model_temperature = temperature
        super().__init__(mutant_creator, debug, debug_folder)

    def __str__(self):
        return (
            f"{self.__class__.__name__}(model={self.identification_model}, "
            f"temperature={self.identification_model_temperature}, "
            f"mutant_creator={self.mutant_creator.__class__.__name__}, "
            f"debug={self.debug}, debug_folder='{self.debug_folder}')"
        )


class ZoneIdentificationModule(LLMMappingModule):
    """LLM-based mutimodal identification module, using regressive box detection."""

    def __str__(self):
        return super().__str__()

    def __init__(
        self,
        *,
        client,
        model,
        temperature=0.3,
        mutant_creator: TexMutantCreator = TexMappingMutantCreator(),
        debug=False,
        debug_folder=".tmp/debug",
    ):
        super().__init__(
            mutant_creator=mutant_creator,
            client=client,
            model=model,
            temperature=temperature,
            debug=debug,
            debug_folder=debug_folder,
        )

    def identify(self, code: str, features: list[str], base_image: Image.Image):
        self.debug_id = str(uuid.uuid4())
        os.mkdir(os.path.join(self.debug_folder, self.debug_id))

        detected_boxes = get_boxes(
            base_image,
            self.identification_client,
            features,
            self.identification_model,
            self.identification_model_temperature,
        )

        if detected_boxes == None:
            logger.warning(
                f"Feature identification failed, using un-identified code, unparseable response"
            )
            return MappedCode(base_image, code, None)

        # create mutants of the code
        mutants = self.mutant_creator.create_mutants(code)

        # Check what has been modified by each mutant
        feature_map: dict[str, list[tuple[CodeImageMapping, float]]] = (
            {}
        )  # mapping between the feature and a list of possible spans of the part of the code of the feature and their "probability" of being the right span

        #computing for each box the disimilarity between original image and mutants ones
        box_image_map = dsim_box(
            detected_boxes, base_image, [mutant.image for mutant in mutants]
        )

        #populating the feature map
        for box, box_mapping in zip(detected_boxes, box_image_map):

            sorted_dsim_mutant_map = [
                (dsim, mutants[mutant_index]) for dsim, mutant_index in box_mapping
            ]

            mappings_for_features: list[tuple[CodeImageMapping, float]] = [
                (CodeImageMapping(mutant.deleted_spans, box["box_2d"]), dsim_value)
                for dsim_value, mutant in sorted_dsim_mutant_map
            ]

            feature_map[box["label"]] = mappings_for_features

        mapped_code = MappedCode(base_image, code, feature_map)
        return mapped_code
