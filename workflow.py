# %%
from workflows import Workflow, step
from workflows.events import (
    Event,
    StartEvent,
    StopEvent,
)

from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from parse import parse_document

load_dotenv(override=True)


class DocumentFlow(Workflow):
    llm = OpenAI(model="gpt-5.1")

    @step
    async def parse_step(self, ev: StartEvent) -> StopEvent:
        markdown_content = await parse_document(ev.file_path)
        return StopEvent(result=markdown_content)


w = DocumentFlow(timeout=60, verbose=False)
result = await w.run(file_path="data/contract-1.pdf")
print(str(result))

# %%
