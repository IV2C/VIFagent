import os
import pickle

from openai import OpenAI
from vif_agent.feature import MappedCode
from vif_agent.modules.edition.edition import LLMAgenticEditionModule
from vif_agent.utils import show_conversation

with open("resources/chimpanzee/p.pickle", "rb") as mp:
    mapped_code: MappedCode = pickle.load(mp)

""" edition_module = LLMEditionModule(
    client=OpenAI(
    ),
    model="gpt-4.1",
    temperature=0.5,
    debug=True,  
) """
""" edition_module = LLMEditionModule(
    client=OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    ),
    model= "meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.5   
) """

edition_module = LLMAgenticEditionModule(
    client=OpenAI(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    model="gemini-2.0-flash",
    temperature=0.3,
    debug=True,
)


res_code = edition_module.customize(
    mapped_code,
    "Make the eyes of the chimpanzee crossed, by making them white and adding black pupils",
)

show_conversation(edition_module.messages)


print(res_code)
