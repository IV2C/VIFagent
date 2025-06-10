import os
from openai import OpenAI

from vif.CodeMapper.mapping import ZoneIdentificationModule
from vif.agent.agent import FeatureAgent
from vif.feature_search.feature_search import SearchModule
from vif.utils.debug_utils import save_conversation
from vif.utils.renderer.tex_renderer import TexRenderer

client = OpenAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

search_module = SearchModule(
    client=client,
    model="gemini-2.0-flash",
)

identification_module = ZoneIdentificationModule(
    client=client,
    model="gemini-2.0-flash",
)

agent: FeatureAgent = FeatureAgent(
    client=client,
    code_renderer=TexRenderer().from_string_to_image,
    model="gemini-2.0-flash",
    search_module=search_module,
    identification_module=identification_module
)


chimp_tex = open("resources/chimpanzee/code.tex").read()
edited_chimp = agent.apply_instruction(
    chimp_tex, "Add a torso to the chimpanzee, in the shape of an ellipse"
)

with open(".tmp/edition/edited_chimp.tex", "w") as edtc:
    edtc.write(edited_chimp)

with open(".tmp/edition/data_conv.tex", "w") as edtc:
    edtc.write(str(agent.conv_data))

save_conversation(agent.messages, ".tmp/edition")
