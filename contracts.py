from pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate


template_string = (
    "You are an expert at extracting rules from legal contracts. "
    "Given the following contract content in markdown format, extract the key rules, constraints, and examples. "
    "Present the extracted information in a structured format.\n\n"
    "Contract Content:\n{markdown_content}"
)


class Rule(BaseModel):
    name: str = Field(description="Name of the rule")
    constraint: str = Field(description="Constraint details of the rules")
    example: str = Field(description="Example illustrating the rule")


class Rules(BaseModel):
    rules: list[Rule] = Field(description="List of extracted rules")


def extract_rules(markdown_content: str) -> list[Rule]:
    llm = OpenAI(model="gpt-5.1")
    prompt = PromptTemplate(template_string)

    response = llm.structured_predict(Rules, prompt, markdown_content=markdown_content)

    return response.rules
