from llama_cloud_services import LlamaParse
from dotenv import load_dotenv

load_dotenv(override=True)

parser = LlamaParse(
    num_workers=4,
    verbose=True,
    language="en",
)


async def parse_document(file_path: str):
    result = await parser.aparse(file_path)
    markdown = await result.aget_markdown()
    return markdown
