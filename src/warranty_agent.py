from dotenv import load_dotenv 
load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0
)


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

    chain = prompt | llm

    response = chain.invoke({
        "equipment": equipment,
        "prediction": prediction
    })

    return response.content