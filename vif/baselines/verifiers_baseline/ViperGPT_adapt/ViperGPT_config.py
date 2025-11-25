import open_clip
import torch
import os
from google import genai
from google.genai import types as genTypes
from openai import Client


class ViperGPTConfig:
    
    ratio_box_area_to_image_area = 0.0
    thresh_clip = 0.6
    # for box detection
    visual_client: genai.Client = genai.Client(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        http_options=genTypes.HttpOptions(api_version="v1alpha"),
    )
    visual_model: str = "gemini-2.5-flash"
    # for simple qa(simple_query function)
    qa_client = Client(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    qa_model = "qwen/qwen3-vl-8b-instruct"
    qa_temperature = 0.7
    # for llm_query
    query_client = Client(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    query_model = "qwen/qwen3-vl-8b-instruct"
    query_temperature = 0.7
    ##clip model
    _clip_model = None
    _clip_preprocess = None
    _clip_tokenizer = None

    @classmethod
    def get_clip_model(cls):
        if cls._clip_model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            cls._clip_model = model.to("cpu").eval()
            cls._clip_preprocess = preprocess
            cls._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return cls._clip_model, cls._clip_preprocess, cls._clip_tokenizer
