from __future__ import annotations
import time
from typing import List
from pydantic import BaseModel

from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.blocks.base import BlockId
from marker.schema.document import Document
from marker.processors.llm import BaseLLMProcessor

class DocumentItem(BaseModel):
    block_id: str
    parent_block_id: str | None

class LLMHierarchySchema(BaseModel):
    document: List[DocumentItem]

class LLMHierarchyProcessor(BaseLLMProcessor):
    image_block_types = (BlockTypes.Picture, BlockTypes.Figure)
    llm_hierarchy_prompt = """
Your task is to analyze the semantic structure of a legal document and output a **flat JSON list**, where each block is represented as an object that includes:

- `block_id`: The original ID of the block.
- `parent_block_id`: A str of ID indicating the **immediate** semantic parent of this block. If the block is a top-level section, this will be None

## Input Description

You will be provided with the identified blocks of a document, where each block is identified by a unique `block_id`. These follow the format `/page/<page_number>/<block_type>/<block_index>`.
You will also be provided with a **snippet** of the text inside of each block that includes the start and end of the block, for context.
Example:
```
/page/0/SectionHeader/0:
Provisions of the Agreement

/page/0/Text/1:
Provision 1: 1st provision

/page/0/Text/2:
Provision a: Sub provision of 1st provision
...
```

## Output Requirements
Your output **must be a single JSON array**. Each element in the array represents a document block with the following structure:

```json
{
    "document": [
        {
            "block_id": "/page/0/SectionHeader/0",
            "parent_block_id": null
        },
        {
            "block_id": "/page/0/Text/1",
            "parent_block_id": "/page/0/SectionHeader/0"
        },
        {
            "block_id": "/page/0/Text/2",
            "parent_block_id": "/page/0/Text/1"
        },
        ...
    ]
}
```

**Key principles for semantic nesting:**
* **Hierarchical Numbering:** Blocks that begin with numbering (e.g., "1.", "1.1", "1.2") or lettering (e.g., "a.", "b.") often indicate nested structure. Use these cues to infer parent-child relationships between blocks.
* **Multi-level Lists:** When list items include sub-items (e.g., bullet points with nested sub-bullets), ensure that sub-items are assigned the correct parent based on the textual content
* **Elements broken across page/column:** A single body of text may sometimes be broken into two elements due to a page/column break. If this is the case, the 2nd element should have the 1st element as its parent
* **Introduction/Preamble:** If a document has a few sections of introductory text at the start (usually on page 0), they should all have the same parent id - Such as the main title or the first element
* **Main Title:** The main title of the document (often a section header) should be considered as the parent for all top-level blocks. In the absence of this, top level blocks may have **no parent**

- It is **crucial** to get the hierarchical nesting in numbered sections correctly
- The top level sections (1, 2, 3 ...) should have a common parent, such as the document title, or no parent at all. It is important to be consistent here, and follow the same for all the top level sections
- Carefully mark text broken into two elements (due to page/column break) as per the rules above.
## Input

"""

    def formatted_block(self, block: Block, document: Document):
        block_raw_text = block.raw_text(document)
        if len(block_raw_text) <= 100:
            formatted_text = block_raw_text
        else:
            formatted_text = block_raw_text[:50] + "..." + block_raw_text[-50:]
        return f"{block.id}:\n{formatted_text}\n\n"

    def unroll_list_groups(self, llm_hierarchy: List[dict], document: Document):
        updated_llm_hierarchy = []
        for hierarchy_dict in llm_hierarchy:
            block_id = BlockId.from_str(hierarchy_dict['block_id'])
            if document.get_block(block_id).block_type != BlockTypes.ListGroup:
                updated_llm_hierarchy.append(hierarchy_dict)
                continue

            parent_id = hierarchy_dict['parent_block_id']
            for list_item_block in document.get_block(block_id).structure_blocks(document):
                updated_llm_hierarchy.append({
                    'block_id': str(list_item_block.id),
                    'parent_block_id': parent_id
                })

        return updated_llm_hierarchy

    def __call__(self, document: Document, **kwargs):
        if not self.use_llm or self.llm_service is None:
            return

        text = "\n"
        for page in document.pages:
            for block in page.structure_blocks(document):
                if block.ignore_for_output or block.block_type in self.image_block_types:
                    continue
                if block.block_type == BlockTypes.ListGroup:
                    for sub_block in block.structure_blocks(document):
                        if sub_block.block_type != BlockTypes.ListItem:
                            continue
                        text += self.formatted_block(sub_block, document)
                else:
                    text += self.formatted_block(block, document)

        response = self.llm_service(self.llm_hierarchy_prompt + text, None, None, LLMHierarchySchema)
        llm_hierarchy = response['document']
        
        document.llm_hierarchy = llm_hierarchy