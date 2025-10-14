ratio_box_area_to_image_area = 0.0
thresh_clip = 0.6

import open_clip
import torch
import os
from google import genai
from google.genai import types as genTypes

visual_client: genai.Client = genai.Client(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    http_options=genTypes.HttpOptions(api_version="v1alpha"),
)
visual_model: str = "gemini-2.5-flash"

##model settings for shape and color detection
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
device = torch.device("cpu")
clip_model = clip_model.to(device)
clip_model.eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")