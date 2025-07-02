import os
from openai import OpenAI

from vif.agent.agent import FeatureAgent
from vif.falcon.edition import OracleEditionModule
from vif.falcon.falcon import Falcon
from vif.falcon.oracle.score_oracle import OracleScoreBoxModule
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

edition_module = OracleEditionModule(
    client=client,
    model="gemini-2.0-flash",
    debug=True
)

identification_module = OracleScoreBoxModule(
    client=client,
    model="gemini-2.0-flash",
)

agent: Falcon = Falcon(
    code_renderer=TexRenderer().from_string_to_image,
    search_module=search_module,
    identification_module=identification_module,
    edition_module=edition_module
)


chimp_tex = open("resources/chimpanzee/code.tex").read()
edited_chimp = agent.apply_instruction(
    chimp_tex, "Add a torso to the chimpanzee, in the shape of an ellipse"
)

with open(".tmp/edition/edited_chimp.tex", "w") as edtc:
    edtc.write(edited_chimp)


