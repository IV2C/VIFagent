from dataclasses import dataclass
import json
import math
import os
import shutil
import uuid

from build.lib.vif_agent.prompt import DETECTION_PROMPT
from vif_agent.modules.identification.identification import IdentificationModule
from vif_agent.modules.identification.utils import get_boxes
from vif_agent.mutation.tex_mutant_creator import (
    TexMappingMutantCreator,
    TexMutantCreator,
)

from vif_agent.feature import CodeImageMapping, MappedCode
from vif_agent.mutation.mutant import TexMutant

from PIL import Image

from vif_agent.utils import adjust_bbox, encode_image, norm_mse

import re
from loguru import logger


class IdentificationMappingModule(IdentificationModule):
    def __init__(
        self,
        mutant_creator: TexMutantCreator = TexMappingMutantCreator(),
        debug: bool = False,
        debug_folder: str = ".tmp/debug",
    ):
        self.mutant_creator = mutant_creator
        super().__init__(debug=debug, debug_folder=debug_folder)

    def map_code(
        self, code: str, features: list[str], base_image: Image.Image
    ) -> MappedCode:
        pass

    def __str__(self):
        return (
            f"{self.__class__.__name__}(mutant_creator={self.mutant_creator.__class__.__name__}, "
            f"debug={self.debug}, debug_folder='{self.debug_folder}')"
        )


class LLMIdentificationModule(IdentificationMappingModule):

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


class BoxIdentificationModule(LLMIdentificationModule):
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
        features = features["features"]

        detected_boxes = get_boxes(
            base_image,
            self.identification_client,
            self.identification_model,
            self.identification_model_temperature,
        )

        if detected_boxes == None:
            logger.warning(
                f"Feature identification failed, using un-identified code, unparseable response"
            )
            return MappedCode(base_image, code, None)

        """DEBUG"""
        if self.debug:
            json.dump(
                detected_boxes,
                open(os.path.join(self.debug_folder, self.debug_id, "boxes.json"), "w"),
            )
        """"""
        # create mutants of the code
        mutants = self.mutant_creator.create_mutants(code)

        # Check what has been modified by each mutant
        feature_map: dict[str, list[tuple[CodeImageMapping, float]]] = (
            {}
        )  # mapping between the feature and a list of possible spans of the part of the code of the feature and their "probability" of being the right span

        """DEBUG"""
        if self.debug:
            base_image.save(
                os.path.join(self.debug_folder, self.debug_id, "base_image.png")
            )
            shutil.rmtree(
                os.path.join(self.debug_folder, self.debug_id, "features/"),
                ignore_errors=True,
            )
            os.mkdir(os.path.join(self.debug_folder, self.debug_id, "features"))
        """"""
        for box in detected_boxes:
            base_image_mask = base_image.crop(box["box_2d"])
            """DEBUG"""
            if self.debug:
                base_image_mask.save(
                    os.path.join(
                        self.debug_folder,
                        self.debug_id,
                        "features/" + box["label"] + ".png",
                    )
                )
            """"""
            cur_mse_map: list[tuple[float, TexMutant]] = []
            for mutant in mutants:
                mutant_image_mask = mutant.image.crop(box["box_2d"])
                cur_mse_map.append(
                    (
                        norm_mse(base_image_mask, mutant_image_mask)
                        / math.prod(base_image_mask.size),
                        mutant,
                    )  # normalized MSE divided by the size of the image, to favoritize small specific features
                )

            sorted_mse_map: list[tuple[float, TexMutant]] = sorted(
                filter(lambda m: m[0] != 0, cur_mse_map),
                key=lambda m: m[0],
                reverse=True,
            )

            mappings_for_features: list[tuple[CodeImageMapping, float]] = [
                (CodeImageMapping(mutant.deleted_spans, box["box_2d"]), mse_value)
                for mse_value, mutant in sorted_mse_map
            ]

            feature_map[box["label"]] = mappings_for_features

            mapped_code = MappedCode(base_image, code, feature_map)
        return mapped_code
