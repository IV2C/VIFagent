import open_clip
import torch

from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline


class ClipSimVerifier(TexVerBaseline):
    def __init__(self, *args, **kwargs):
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self._clip_model = model.to("cpu").eval()
        self._clip_preprocess = preprocess
        self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

        super().__init__(*args, **kwargs)
    def get_config_metadata(self):
        return {
            "name": "ClipSimVerifier",
            "model_name":"ViT-B-32",
            "pretrained_name":"laion2b_s34b_b79k"
        }
    def assess_customization(self, ver_eval_input):
        perfect = self._clip_preprocess(
            ver_eval_input.theoretical_perfect_image
        ).unsqueeze(0)
        solution = self._clip_preprocess(
            ver_eval_input.initial_solution_image
        ).unsqueeze(0)

        with torch.no_grad(), torch.autocast("cuda"):
            perfect_image_features = self._clip_model.encode_image(perfect)
            solution_image_features = self._clip_model.encode_image(solution)
            perfect_image_features /= perfect_image_features.norm(dim=-1, keepdim=True)
            solution_image_features /= solution_image_features.norm(dim=-1, keepdim=True)

            similarity = (solution_image_features @ perfect_image_features.T).squeeze()
        
        ver_eval_input.classified_score = similarity
        
        return ver_eval_input
        
        