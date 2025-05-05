from vif_agent.agent import VifAgent
from openai import OpenAI
from vif_agent.modules.identification.identification import BoxIdentificationModule
from vif_agent.renderer.tex_renderer import TexRenderer
import os
import pickle


def test_comment():

    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    identification_module = BoxIdentificationModule(
        client=OpenAI(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        model="gemini-2.0-flash",
        temperature=0.3,
        debug=True
    )

    search_client = OpenAI(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    agent = VifAgent(
        TexRenderer().from_string_to_image,
        client=client,
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        search_client=search_client,
        search_model="gemini-2.0-flash",
        search_model_temperature=0.0,
        identification_module=identification_module,
        debug=True,
        clarify_instruction=False,
    )

    chimp_tex = open("plane.tex").read()
    mapped_code = agent.identify_features(chimp_tex)

    with open("resources/plane/p.pickle", "wb") as mp:
        pickle.dump(mapped_code, mp)
    print(mapped_code.feature_map.keys())

    # commented_code = mapped_code.get_commented_code()
    # modified_code = agent.apply_instruction(dog_tex, "Make the eyes of the chimpanzee crossed, by making them white and adding black pupils")

    # with open("commented_chimpanzee.tex", "w") as md_chimp:
    #    md_chimp.write(commented_code)


def main():
    test_comment()


if __name__ == "__main__":
    main()
