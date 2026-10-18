import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

_llm = None


def get_llm():
    # Created on first use, so importing this module does not need an API key.
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0
        )
    return _llm


prompt = ChatPromptTemplate.from_template("""
You are a predictive maintenance and warranty assistant.

Use the equipment information and machine-learning prediction
to provide a practical maintenance recommendation.

Equipment information:
{equipment}

ML prediction:
{prediction}

Provide:

Risk Assessment:
<explain the risk>

Likely Failure Concern:
<what component or condition should be investigated>

Recommended Parts:
<parts or components to inspect>

Warranty / Service Recommendation:
<recommended service or warranty action>

Priority:
<LOW, MEDIUM, HIGH, or URGENT>

Important:
Do not claim that a specific component has definitely failed.
The ML model predicts risk, not confirmed failure.
""")


def generate_recommendation(equipment, prediction):

    chain = prompt | get_llm()

    response = chain.invoke({
        "equipment": equipment,
        "prediction": prediction
    })

    return response.content