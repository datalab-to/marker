import json

from pydantic import BaseModel
from typing import Annotated, Optional, List

from marker.extractors import BaseExtractor
from marker.extractors.page import PageExtractionSchema
from marker.logger import get_logger

logger = get_logger()


class DocumentExtractionSchema(BaseModel):
    analysis: str
    document_json: str


class DocumentExtractor(BaseExtractor):
    """
    An extractor that combines data from across all pages.
    """

    page_schema: Annotated[
        str,
        "The JSON schema to be extracted from the page.",
    ] = ""

    page_extraction_prompt = """You are an expert document analyst who reads documents and pulls data out in JSON format. You will receive your detailed notes from all the pages of a document, and a JSON schema that we want to extract from the document. Your task is to extract all the information properly into the JSON schema.

Some notes:
- The schema may contain a single object to extract from the entire document, or an array of objects. 
- The schema may contain nested objects, arrays, and other complex structures.

Some guidelines:
- Some entities will span multiple pages, so make sure to consult your notes thoroughly.
- In the case of potential conflicting values, pull out the values you have the most confidence in, from your notes.
- If you cannot find a value for a field, leave it blank in the JSON.

**Instructions:**
1. Analyze your provided notes.
2. Analyze the JSON schema.
3. Write a detailed analysis of the notes, and the associated values in the schema.  Make sure to reference which page each piece of information comes from.
4. Write the output in the JSON schema format, ensuring all required fields are filled out.  Output only the json data, without any additional text or formatting.

**Example:**
Input:

Detailed Notes
Page 0
On this page, I see a table with car makes and sales. The makes are Honda and Toyota, with sales of 100 and 200 respectively. The color is not present in the table, so I will leave it blank in the JSON.  That information may be present on another page.  Some JSON snippets I may find useful later are:
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

Page 1
I see a table that contains 2 rows, and has a color header.  The first row has the color red, and the second row has the color blue.  Here are some useful snippets:

Schema

```json
{'$defs': {'Cars': {'properties': {'make': {'title': 'Make', 'type': 'string'}, 'sales': {'title': 'Sales', 'type': 'integer'}, 'color': {'title': 'Color', 'type': 'string'}}, 'required': ['make', 'sales', 'color'], 'title': 'Cars', 'type': 'object'}}, 'properties': {'cars': {'items': {'$ref': '#/$defs/Cars'}, 'title': 'Cars', 'type': 'array'}}, 'required': ['cars'], 'title': 'CarsList', 'type': 'object'}
```

Output:

Analysis: From the notes, it looks like the information I need is in a table that spans 2 pages.  The first page has the makes and sales, while the second page has the colors.  I will combine this information into the JSON schema.
JSON

{
    "cars": [
        {
            "make": "Honda",
            "sales": 100,
            "color": "red"
        },
        {
            "make": "Toyota",
            "sales": 200,
            "color": "blue"
        }
    ]
}

**Input:**

Detailed Notes
{{document_notes}}

Schema
```json
{{schema}}
```
"""

    def assemble_document_notes(self, page_notes: List[PageExtractionSchema]) -> str:
        notes = ""
        for i, page_schema in enumerate(page_notes):
            if not page_notes:
                continue
            notes += f"Page {i + 1}\n{page_schema.detailed_notes}\n\n"
        return notes.strip()

    def __call__(
        self,
        page_notes: List[PageExtractionSchema],
        **kwargs,
    ) -> Optional[DocumentExtractionSchema]:
        if not self.page_schema:
            raise ValueError(
                "Page schema must be defined for structured extraction to work."
            )

        prompt = self.page_extraction_prompt.replace(
            "{{document_notes}}", self.assemble_document_notes(page_notes)
        ).replace("{{schema}}", json.dumps(self.page_schema))
        response = self.llm_service(prompt, None, None, DocumentExtractionSchema)

        logger.debug(f"Document extraction response: {response}")

        if not response or any(
            [
                key not in response
                for key in [
                    "analysis",
                    "document_json",
                ]
            ]
        ):
            return None

        json_data = response["document_json"].strip().lstrip("```json").rstrip("```")

        return DocumentExtractionSchema(
            analysis=response["analysis"], document_json=json_data
        )
