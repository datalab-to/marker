from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, List, Tuple, Literal

from pydantic import BaseModel
from tqdm import tqdm
from PIL import Image

from marker.output import json_to_html
from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block, TableCell
from marker.schema.document import Document
from marker.logger import get_logger

logger = get_logger()

class LLMTableMergeProcessor(BaseLLMComplexBlockProcessor):
    block_types: Annotated[
        Tuple[BlockTypes],
        "The block types to process.",
    ] = (BlockTypes.Table, BlockTypes.TableOfContents)
    table_height_threshold: Annotated[
        float,
        "The minimum height ratio relative to the page for the first table in a pair to be considered for merging.",
    ] = 0.6
    table_start_threshold: Annotated[
        float,
        "The maximum percentage down the page the second table can start to be considered for merging."
    ] = 0.2
    vertical_table_height_threshold: Annotated[
        float,
        "The height tolerance for 2 adjacent tables to be merged into one."
    ] = 0.25
    vertical_table_distance_threshold: Annotated[
        int,
        "The maximum distance between table edges for adjacency."
    ] = 20
    horizontal_table_width_threshold: Annotated[
        float,
        "The width tolerance for 2 adjacent tables to be merged into one."
    ] = 0.25
    horizontal_table_distance_threshold: Annotated[
        int,
        "The maximum distance between table edges for adjacency."
    ] = 10
    column_gap_threshold: Annotated[
        int,
        "The maximum gap between columns to merge tables"
    ] = 50
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False
    no_merge_tables_across_pages: Annotated[
        bool,
        "Whether to disable merging tables across pages and keep page delimiters.",
    ] = False
    table_merge_prompt: Annotated[
        str,
        "The prompt to use for rewriting text.",
        "Default is a string containing the Gemini rewriting prompt."
    ] = """You're a text correction expert specializing in accurately reproducing tables from PDFs.
You'll receive two images of tables from successive pages of a PDF.  Table 1 is from the first page, and Table 2 is from the second page.  Both tables may actually be part of the same larger table. Your job is to decide if Table 2 should be merged with Table 1, and how they should be joined.  The should only be merged if they're part of the same larger table, and Table 2 cannot be interpreted without merging.

You'll specify your judgement in json format - first whether Table 2 should be merged with Table 1, then the direction of the merge, either `bottom` or `right`.  A bottom merge means that the rows of Table 2 are joined to the rows of Table 1. A right merge means that the columns of Table 2 are joined to the columns of Table 1.  (bottom merge is equal to np.vstack, right merge is equal to np.hstack)

Table 2 should be merged at the bottom of Table 1 if Table 2 has no headers, and the rows have similar values, meaning that Table 2 continues Table 1. Table 2 should be merged to the right of Table 1 if each row in Table 2 matches a row in Table 1, meaning that Table 2 contains additional columns that augment Table 1.

Only merge Table 1 and Table 2 if Table 2 cannot be interpreted without merging.  Only merge Table 1 and Table 2 if you can read both images properly.

**Instructions:**
1. Carefully examine the provided table images.  Table 1 is the first image, and Table 2 is the second image.
2. Examine the provided html representations of Table 1 and Table 2.
3. Write a description of Table 1.
4. Write a description of Table 2.
5. Analyze whether Table 2 should be merged into Table 1, and write an explanation.
6. Output your decision on whether they should be merged, and merge direction.
**Example:**
Input:
Table 1
```html
<table>
    <tr>
        <th>Name</th>
        <th>Age</th>
        <th>City</th>
        <th>State</th>
    </tr>
    <tr>
        <td>John</td>
        <td>25</td>
        <td>Chicago</td>
        <td>IL</td>
    </tr>
```
Table 2
```html
<table>
    <tr>
        <td>Jane</td>
        <td>30</td>
        <td>Los Angeles</td>
        <td>CA</td>
    </tr>
```
Output:
```json
{
    "table1_description": "Table 1 has 4 headers, and 1 row.  The headers are Name, Age, City, and State.",
    "table2_description": "Table 2 has no headers, but the values appear to represent a person's name, age, city, and state.",
    "explanation": "The values in Table 2 match the headers in Table 1, and Table 2 has no headers. Table 2 should be merged to the bottom of Table 1.",
    "merge": "true",
    "direction": "bottom"
}
```
**Input:**
Table 1
```html
{{table1}}
Table 2
```html
{{table2}}
```
"""

    @staticmethod
    def get_row_count(cells: List[TableCell]):
        if not cells:
            return 0

        max_rows = None
        for col_id in set([cell.col_id for cell in cells]):
            col_cells = [cell for cell in cells if cell.col_id == col_id]
            rows = 0
            for cell in col_cells:
                rows += cell.rowspan
            if max_rows is None or rows > max_rows:
                max_rows = rows
        return max_rows

    @staticmethod
    def get_column_count(cells: List[TableCell]):
        if not cells:
            return 0

        max_cols = None
        for row_id in set([cell.row_id for cell in cells]):
            row_cells = [cell for cell in cells if cell.row_id == row_id]
            cols = 0
            for cell in row_cells:
                cols += cell.colspan
            if max_cols is None or cols > max_cols:
                max_cols = cols
        return max_cols

    def rewrite_blocks(self, document: Document):
        # Skip table merging if disabled via config
        if self.no_merge_tables_across_pages:
            logger.info("Skipping table merging across pages due to --no_merge_tables_across_pages flag")
            return
            
        pbar = tqdm(desc=f"{self.__class__.__name__} running", disable=self.disable_tqdm)
        table_runs = []
        table_run = []
        prev_block = None
        prev_page_block_count = None
        for page in document.pages:
            page_blocks = page.contained_blocks(document, self.block_types)
            for block in page_blocks:
                merge_condition = False
                if prev_block is not None:
                    prev_cells = prev_block.contained_blocks(document, (BlockTypes.TableCell,))
                    curr_cells = block.contained_blocks(document, (BlockTypes.TableCell,))
                    row_match = abs(self.get_row_count(prev_cells) - self.get_row_count(curr_cells)) < 5, # Similar number of rows
                    col_match = abs(self.get_column_count(prev_cells) - self.get_column_count(curr_cells)) < 2

                    subsequent_page_table = all([
                        prev_block.page_id == block.page_id - 1, # Subsequent pages
                        max(prev_block.polygon.height / page.polygon.height,
                            block.polygon.height / page.polygon.height) > self.table_height_threshold, # Take up most of the page height
                            (len(page_blocks) == 1 or prev_page_block_count == 1), # Only table on the page
                            (row_match or col_match)
                        ])

                    same_page_vertical_table = all([
                        prev_block.page_id == block.page_id, # On the same page
                        (1 - self.vertical_table_height_threshold) < prev_block.polygon.height / block.polygon.height < (1 + self.vertical_table_height_threshold), # Similar height
                        abs(block.polygon.x_start - prev_block.polygon.x_end) < self.vertical_table_distance_threshold, # Close together in x
                        abs(block.polygon.y_start - prev_block.polygon.y_start) < self.vertical_table_distance_threshold, # Close together in y
                        row_match
                    ])

                    same_page_horizontal_table = all([
                        prev_block.page_id == block.page_id, # On the same page
                        (1 - self.horizontal_table_width_threshold) < prev_block.polygon.width / block.polygon.width < (1 + self.horizontal_table_width_threshold), # Similar width
                        abs(block.polygon.y_start - prev_block.polygon.y_end) < self.horizontal_table_distance_threshold, # Close together in y
                        abs(block.polygon.x_start - prev_block.polygon.x_start) < self.horizontal_table_distance_threshold, # Close together in x
                        col_match
                    ])

                    same_page_new_column = all([
                        prev_block.page_id == block.page_id, # On the same page
                        abs(block.polygon.x_start - prev_block.polygon.x_end) < self.column_gap_threshold,
                        block.polygon.y_start < prev_block.polygon.y_end,
                        block.polygon.width * (1 - self.vertical_table_height_threshold) < prev_block.polygon.width  < block.polygon.width * (1 + self.vertical_table_height_threshold), # Similar width
                        col_match
                    ])
                    merge_condition = any([subsequent_page_table, same_page_vertical_table, same_page_new_column, same_page_horizontal_table])

                if prev_block is not None and merge_condition:
                    if prev_block not in table_run:
                        table_run.append(prev_block)
                    table_run.append(block)
                else:
                    if table_run:
                        table_runs.append(table_run)
                    table_run = []
                prev_block = block
            prev_page_block_count = len(page_blocks)

        if table_run:
            table_runs.append(table_run)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            for future in as_completed([
                executor.submit(self.process_rewriting, document, blocks)
                for blocks in table_runs
            ]):
                future.result()  # Raise exceptions if any occurred
                pbar.update(1)

        pbar.close()

    def process_rewriting(self, document: Document, blocks: List[Block]):
        if len(blocks) < 2:
            # Can't merge single tables
            return

        start_block = blocks[0]
        for i in range(1, len(blocks)):
            curr_block = blocks[i]
            children = start_block.contained_blocks(document, (BlockTypes.TableCell,))
            children_curr = curr_block.contained_blocks(document, (BlockTypes.TableCell,))
            if not children or not children_curr:
                # Happens if table/form processors didn't run
                break

            start_image = start_block.get_image(document, highres=False)
            curr_image = curr_block.get_image(document, highres=False)
            start_html = json_to_html(start_block.render(document))
            curr_html = json_to_html(curr_block.render(document))

            prompt = self.table_merge_prompt.replace("{{table1}}", start_html).replace("{{table2}}", curr_html)

            response = self.llm_service(
                prompt,
                [start_image, curr_image],
                curr_block,
                MergeSchema,
            )

            if not response or ("direction" not in response or "merge" not in response):
                curr_block.update_metadata(llm_error_count=1)
                break

            merge = response["merge"]

            # The original table is okay
            if "true" not in merge:
                start_block = curr_block
                continue

            # Merge the cells and images of the tables
            direction = response["direction"]
            if not self.validate_merge(children, children_curr, direction):
                start_block = curr_block
                continue

            merged_image = self.join_images(start_image, curr_image, direction)
            merged_cells = self.join_cells(children, children_curr, direction)
            curr_block.structure = []
            start_block.structure = [b.id for b in merged_cells]
            start_block.lowres_image = merged_image

    def validate_merge(self, cells1: List[TableCell], cells2: List[TableCell], direction: Literal['right', 'bottom'] = 'right'):
        if direction == "right":
            # Check if the number of rows is the same
            cells1_row_count = self.get_row_count(cells1)
            cells2_row_count = self.get_row_count(cells2)
            return abs(cells1_row_count - cells2_row_count) < 5
        elif direction == "bottom":
            # Check if the number of columns is the same
            cells1_col_count = self.get_column_count(cells1)
            cells2_col_count = self.get_column_count(cells2)
            return abs(cells1_col_count - cells2_col_count) < 2


    def join_cells(self, cells1: List[TableCell], cells2: List[TableCell], direction: Literal['right', 'bottom'] = 'right') -> List[TableCell]:
        if direction == 'right':
            # Shift columns right
            col_count = self.get_column_count(cells1)
            for cell in cells2:
                cell.col_id += col_count
            new_cells = cells1 + cells2
        else:
            # Shift rows up
            row_count = self.get_row_count(cells1)
            for cell in cells2:
                cell.row_id += row_count
            new_cells = cells1 + cells2
        return new_cells

    @staticmethod
    def join_images(image1: Image.Image, image2: Image.Image, direction: Literal['right', 'bottom'] = 'right') -> Image.Image:
        # Get dimensions
        w1, h1 = image1.size
        w2, h2 = image2.size

        if direction == 'right':
            new_height = max(h1, h2)
            new_width = w1 + w2
            new_img = Image.new('RGB', (new_width, new_height), 'white')
            new_img.paste(image1, (0, 0))
            new_img.paste(image2, (w1, 0))
        else:
            new_width = max(w1, w2)
            new_height = h1 + h2
            new_img = Image.new('RGB', (new_width, new_height), 'white')
            new_img.paste(image1, (0, 0))
            new_img.paste(image2, (0, h1))
        return new_img


class MergeSchema(BaseModel):
    table1_description: str
    table2_description: str
    explanation: str
    merge: Literal["true", "false"]
    direction: Literal["bottom", "right"]