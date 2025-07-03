import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
from typing import Annotated, Optional, List

from tqdm import tqdm

from marker.extractors import BaseExtractor
from marker.logger import get_logger

logger = get_logger()


class PageExtractionSchema(BaseModel):
    description: str
    detailed_notes: str


class PageExtractor(BaseExtractor):
    """
    An extractor that pulls data from a single page.
    """

    extraction_page_chunk_size: Annotated[
        int, "The number of pages to chunk together for extraction."
    ] = 3

    page_schema: Annotated[
        str,
        "The JSON schema to be extracted from the page.",
    ] = ""

    page_extraction_prompt = """You are an expert document analyst who reads documents and pulls data out in JSON format. You will receive the markdown representation of a document page, and a JSON schema that we want to extract from the document. Your task is to write detailed notes on this page, so that when you look at all your notes from across the document, you can fill in the schema.
    
Some notes:
- The schema may contain a single object to extract from the entire document, or an array of objects. 
- The schema may contain nested objects, arrays, and other complex structures.

Some guidelines:
- Write very thorough notes, and include specific JSON snippets that can be extracted from the page.
- You may need information from prior or subsequent pages to fully fill in the schema, so make sure to write detailed notes that will let you join entities across pages later on.
- Estimate your confidence in the values you extract, so you can reconstruct the JSON later when you only have your notes.
- Some tables and other data structures may continue on a subsequent page, so make sure to store the positions that data comes from where appropriate.

**Instructions:**
1. Analyze the provided markdown representation of the page.
2. Analyze the JSON schema.
3. Write a short description of the fields in the schema, and the associated values in the markdown.
4. Write detailed notes on the page, including any values that can be extracted from the markdown.  Include snippets of JSON that can be extracted from the page where possible.

**Example:**
Input:

Markdown
```markdown
| Make   | Sales |
|--------|-------|
| Honda  | 100   |
| Toyota | 200   |
```

Schema

```json
{'$defs': {'Cars': {'properties': {'make': {'title': 'Make', 'type': 'string'}, 'sales': {'title': 'Sales', 'type': 'integer'}, 'color': {'title': 'Color', 'type': 'string'}}, 'required': ['make', 'sales', 'color'], 'title': 'Cars', 'type': 'object'}}, 'properties': {'cars': {'items': {'$ref': '#/$defs/Cars'}, 'title': 'Cars', 'type': 'array'}}, 'required': ['cars'], 'title': 'CarsList', 'type': 'object'}
```

Output:

Description: The schema has a list of cars, each with a make, sales, and color. The image and markdown contain a table with 2 cars: Honda with 100 sales and Toyota with 200 sales. The color is not present in the table.
Detailed Notes: On this page, I see a table with car makes and sales. The makes are Honda and Toyota, with sales of 100 and 200 respectively. The color is not present in the table, so I will leave it blank in the JSON.  That information may be present on another page.  Some JSON snippets I may find useful later are:
```json
{
    "make": "Honda",
    "sales": 100,
}
```
```json
{
    "make": "Toyota",
    "sales": 200,
}
```

Honda is the first row in the table, and Toyota is the second row.  Make is the first column, and sales is the second.

**Input:**

Markdown
```markdown
{{page_md}}
```

Schema
```json
{{schema}}
```
"""

    def chunk_page_markdown(self, page_markdown: List[str]) -> List[str]:
        """
        Chunk the page markdown into smaller pieces for processing.
        """

        chunks = []
        for i in range(0, len(page_markdown), self.extraction_page_chunk_size):
            chunk = page_markdown[i : i + self.extraction_page_chunk_size]
            chunks.append("\n\n".join(chunk))

        return chunks

    def inference_single_chunk(
        self, page_markdown: str
    ) -> Optional[PageExtractionSchema]:
        prompt = self.page_extraction_prompt.replace(
            "{{page_md}}", page_markdown
        ).replace("{{schema}}", json.dumps(self.page_schema))
        response = self.llm_service(prompt, None, None, PageExtractionSchema)
        logger.debug(f"Page extraction response: {response}")

        if not response or any(
            [
                key not in response
                for key in [
                    "description",
                    "detailed_notes",
                ]
            ]
        ):
            return None

        return PageExtractionSchema(
            description=response["description"],
            detailed_notes=response["detailed_notes"],
        )

    def __call__(
        self,
        page_markdown: List[str],
        **kwargs,
    ) -> List[PageExtractionSchema]:
        if not self.page_schema:
            raise ValueError(
                "Page schema must be defined for structured extraction to work."
            )

        chunks = self.chunk_page_markdown(page_markdown)
        results = []
        pbar = tqdm(
            desc="Running page extraction",
            disable=self.disable_tqdm,
            total=len(chunks),
        )

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            for future in [
                executor.submit(self.inference_single_chunk, chunk) for chunk in chunks
            ]:
                results.append(future.result())  # Raise exceptions if any occurred
                pbar.update(1)

        pbar.close()
        return results
