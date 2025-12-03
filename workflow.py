# %%
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    StartEvent,
    StopEvent,
    Context,
)

from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from parse import parse_document
from contracts import extract_rules, Rule
from invoice import validate_invoice
from typing import List
from pprint import pprint

load_dotenv(override=True)


class ContractParsedEvent(Event):
    markdown_content: str


class InvoiceParsedEvent(Event):
    markdown_content: str


class RulesExtractedEvent(Event):
    rules: List[Rule]


class DocumentFlow(Workflow):
    llm = OpenAI(model="gpt-5.1")

    @step
    async def parse_contract_step(self, ev: StartEvent) -> ContractParsedEvent:
        markdown_content = await parse_document(ev.contract_path)
        return ContractParsedEvent(markdown_content=markdown_content)

    @step
    async def parse_invoice_step(self, ev: StartEvent) -> InvoiceParsedEvent:
        markdown_content = await parse_document(ev.invoice_path)
        return InvoiceParsedEvent(markdown_content=markdown_content)

    @step
    async def extract_rules_step(self, ev: ContractParsedEvent) -> RulesExtractedEvent:
        rules = extract_rules(ev.markdown_content)
        return RulesExtractedEvent(rules=rules)

    @step
    async def validate_invoice_step(
        self, ctx: Context, ev: InvoiceParsedEvent | RulesExtractedEvent
    ) -> StopEvent | None:
        ready = ctx.collect_events(ev, [InvoiceParsedEvent, RulesExtractedEvent])

        # If not all events are collected, return None (wait for next event)
        if ready is None:
            return None

        invoice_ev, rules_ev = ready
        compliance_result = validate_invoice(
            invoice_ev.markdown_content, rules_ev.rules
        )

        return StopEvent(result=compliance_result)


w = DocumentFlow(timeout=120, verbose=True)
result = await w.run(
    contract_path="data/contract-1.pdf", invoice_path="data/invoice-1-3.pdf"
)
pprint(str(result))

# %%
