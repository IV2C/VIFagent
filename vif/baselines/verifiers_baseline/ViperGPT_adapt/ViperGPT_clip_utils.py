from typing import Union
import torch
from vif.baselines.verifiers_baseline.ViperGPT_adapt import ViperGPT_config
import torch.nn.functional as F


def score(image, category, negative_categories):

    image = ViperGPT_config.preprocess(image).unsqueeze(0)

    prompt_prefix = "photo of "
    prompt = prompt_prefix + prompt

    negative_text_features = clip_negatives(prompt_prefix, negative_categories)

    text = ViperGPT_config.clip_tokenizer([prompt])

    image_features = ViperGPT_config.clip_model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    pos_text_features = ViperGPT_config.clip_model.encode_text(text)
    pos_text_features /= pos_text_features.norm(dim=-1, keepdim=True)

    text_features = torch.concat([pos_text_features, negative_text_features], axis=0)

    # run competition where we do a binary classification
    # between the positive and all the negatives, then take the mean
    sim = (100.0 * image_features @ text_features.T).squeeze(dim=0)

    res = F.softmax(
        torch.cat(
            (sim[0].broadcast_to(1, sim.shape[0] - 1), sim[1:].unsqueeze(0)), dim=0
        ),
        dim=0,
    )[0].mean()
    return res


def clip_negatives(prompt_prefix, negative_categories=None):
    if negative_categories is None:
        with open(
            "vif/baselines/verifiers_baseline/ViperGPT_adapt/useful_lists/random_negatives.txt"
        ) as f:
            negative_categories = [x.strip() for x in f.read().split()]
    # negative_categories = negative_categories[:1000]
    # negative_categories = ["a cat", "a lamp"]
    negative_categories = [prompt_prefix + x for x in negative_categories]
    negative_tokens = ViperGPT_config.clip_tokenizer(negative_categories)

    negative_text_features = ViperGPT_config.clip_model.encode_text(negative_tokens)
    negative_text_features /= negative_text_features.norm(dim=-1, keepdim=True)

    return negative_text_features


def classify(
    image: torch.Tensor, categories: list[str], return_index=True
):
    
    prompt_prefix = "photo of "
    categories = [prompt_prefix + x for x in categories]
    categories = ViperGPT_config.clip_tokenizer(categories)

    text_features = ViperGPT_config.clip_model.encode_text(categories)
    text_features = F.normalize(text_features, dim=-1)

    image_features = self.model.encode_image(image_clip)
    image_features = F.normalize(image_features, dim=-1)

    if image_clip.shape[0] == 1:
        # get category from image
        softmax_arg = image_features @ text_features.T  # 1 x n
    else:
        if is_list:
            # get highest category-image match with n images and n corresponding categories
            softmax_arg = (
                (image_features @ text_features.T).diag().unsqueeze(0)
            )  # n x n -> 1 x n
        else:
            softmax_arg = image_features @ text_features.T

    similarity = (100.0 * softmax_arg).softmax(dim=-1).squeeze(0)
    if not return_index:
        return similarity
    else:
        result = torch.argmax(similarity, dim=-1)
        if result.shape == ():
            result = result.item()
        return result
