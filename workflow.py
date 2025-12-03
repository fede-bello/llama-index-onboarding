# %%
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    StartEvent,
    StopEvent,
)

from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from parse import parse_document
from contracts import get_rules_schema

load_dotenv(override=True)


class ParseEvent(Event):
    markdown_content: str


class DocumentFlow(Workflow):
    llm = OpenAI(model="gpt-5.1")

    @step
    async def parse_step(self, ev: StartEvent) -> ParseEvent:
        markdown_content = await parse_document(ev.file_path)
        return ParseEvent(markdown_content=markdown_content)

    @step
    async def extract_rules(self, ev: ParseEvent) -> StopEvent:
        rules = get_rules_schema(ev.markdown_content)
        return StopEvent(result=rules)


w = DocumentFlow(timeout=60, verbose=False)
result = await w.run(file_path="data/contract-1.pdf")
print(str(result))

# %%
