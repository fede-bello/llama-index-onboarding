from typing import List
from pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from contracts import Rule

template_string = (
    "You are an expert at auditing invoices against contract rules. "
    "Given the following invoice content in markdown format and a list of rules, "
    "determine if the invoice is compliant with the rules. "
    "If there are violations, identify the rule, the location in the invoice, and the reason. "
    "IMPORTANT: Only include actual violations in the list. Do not include rules that were followed correctly. "
    "If a rule is satisfied, do not mention it in the violations list.\n\n"
    "Rules:\n{rules_content}\n\n"
    "Invoice Content:\n{invoice_content}"
)


class Violation(BaseModel):
    rule_name: str = Field(description="The name of the rule that was violated")
    location: str = Field(
        description="The specific part or line item in the invoice where the violation occurred"
    )
    reason: str = Field(
        description="Explanation of why the invoice is not compliant with the rule"
    )


class ComplianceResult(BaseModel):
    is_compliant: bool = Field(
        description="Whether the invoice is fully compliant with all rules"
    )
    violations: List[Violation] = Field(description="List of violations found, if any")


def validate_invoice(invoice_markdown: str, rules: List[Rule]) -> ComplianceResult:
    llm = OpenAI(model="gpt-5.1", reasoning_effort="high")
    prompt = PromptTemplate(template_string)

    # Format rules for the prompt
    rules_text = "\n".join(
        [
            f"- {rule.name}: {rule.constraint} (Example: {rule.example})"
            for rule in rules
        ]
    )

    response = llm.structured_predict(
        ComplianceResult,
        prompt,
        rules_content=rules_text,
        invoice_content=invoice_markdown,
    )

    return response
