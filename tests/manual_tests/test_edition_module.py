import os
import pickle

from openai import OpenAI
from vif_agent.feature import MappedCode
from vif_agent.modules.edition.edition import LLMEditionModule


with open("resources/chimpanzee/p.pickle", "rb") as mp:
        mapped_code:MappedCode = pickle.load(mp)
        
edition_module = LLMEditionModule(
    client=OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    ),
    model= "meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.5   
)

res_code = edition_module.customize(mapped_code,"Add a torso to the chimpanzee, in the shape of an ellipse")

print(res_code)