import os
import pickle
from openai import OpenAI

from vif_agent.agent import VifAgent
from vif_agent.modules.edition.edition import LLMAgenticEditionModule
from vif_agent.modules.identification.identification import BoxIdentificationModule
from vif_agent.renderer.tex_renderer import TexRenderer


edition_module = LLMAgenticEditionModule(
    client=OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    ),
    model= "meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.5   
)


identification_module = BoxIdentificationModule(
    client=OpenAI(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    model="gemini-2.0-flash",
    temperature=0.3,
    debug=True,
)

search_client = OpenAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

agent = VifAgent(
    code_renderer=TexRenderer().from_string_to_image,
    edition_module=edition_module,
    search_client=search_client,
    search_model="gemini-2.0-flash",
    search_model_temperature=0.0,
    identification_module=identification_module,
    debug=True,
    clarify_instruction=False,
)

chimp_tex = open("resources/chimpanzee/code.tex").read()
edited_chimp = agent.apply_instruction(chimp_tex,"Add a torso to the chimpanzee, in the shape of an ellipse")

with open(".tmp/edition/edited_chimp.tex", "w") as edtc:
    edtc.write(edited_chimp)

with open(".tmp/edition/data_conv.tex", "w") as edtc:
    edtc.write(agent.edition_module.conv_data)

with open(".tmp/edition/full_conv.tex", "w") as edtc:
    edtc.write(agent.edition_module.messages)
