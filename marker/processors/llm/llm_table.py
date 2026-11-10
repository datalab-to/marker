from typing import Annotated, List, Tuple

from bs4 import BeautifulSoup
from PIL import Image
from marker.logger import get_logger
from pydantic import BaseModel

from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block, TableCell, Table
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox

logger = get_logger()


class LLMTableProcessor(BaseLLMComplexBlockProcessor):
    block_types: Annotated[
        Tuple[BlockTypes],
        "The block types to process.",
    ] = (BlockTypes.Table, BlockTypes.TableOfContents)
    max_rows_per_batch: Annotated[
        int,
        "If the table has more rows than this, chunk the table. (LLMs can be inaccurate with a lot of rows)",
    ] = 60
    max_table_rows: Annotated[
        int,
        "The maximum number of rows in a table to process with the LLM processor.  Beyond this will be skipped.",
    ] = 175
    rotation_max_wh_ratio: Annotated[
        float,
        "The maximum width/height ratio for table cells for a table to be considered rotated.",
    ] = 0.6
    max_table_iterations: Annotated[
        int,
        "The maximum number of iterations to attempt rewriting a table.",
    ] = 2
    table_rewriting_prompt: Annotated[
        str,
        "The prompt to use for rewriting text.",
        "Default is a string containing the Gemini rewriting prompt.",
    ] = """You are a text correction expert specializing in accurately reproducing text from images.
You will receive an image and an html representation of the table in the image.
Your task is to correct any errors in the html representation.  The html representation should be as faithful to the original table image as possible.  The table image may be rotated, but ensure the html representation is not rotated.  Make sure to include HTML for the full table, including the opening and closing table tags.

Some guidelines:
- Reproduce the original values from the image as faithfully as possible.  
- There may be stray characters in the html representation that don't match the image - fix these.
- Ensure column headers match the correct column values.
- If you see any inline math in a table cell, fence it with the <math> tag.  Block math should be fenced with <math display="block">.
- Replace any images in table cells with a description, like "Image: [description]".
- Only use the tags th, td, tr, br, span, sup, sub, i, b, math, and table.  Only use the attributes display, style, colspan, and rowspan if necessary.  You can use br to break up text lines in cells.
- Make sure the columns and rows match the image faithfully, and are easily readable and interpretable by a human.

**Instructions:**
1. Carefully examine the provided text block image.
2. Analyze the html representation of the table.
3. Write a comparison of the image and the html representation, paying special attention to the column headers matching the correct column values.
4. If the html representation is completely correct, or you cannot read the image properly, then write "No corrections needed."  If the html representation has errors, generate the corrected html representation.  Output only either the corrected html representation or "No corrections needed."
5. If you made corrections, analyze your corrections against the original image, and provide a score from 1-5, indicating how well the corrected html matches the image, with 5 being perfect.
**Example:**
Input:
```html
<table>
    <tr>
        <th>First Name</th>
        <th>Last Name</th>
        <th>Age</th>
    </tr>
    <tr>
        <td>John</td>
        <td>Doe</td>
    </tr>
</table>
```
Output:
comparison: The image shows a table with 2 rows and 3 columns.  The text and formatting of the html table matches the image.  The column headers match the correct column values.
```html
No corrections needed.
```
analysis: I did not make any corrections, as the html representation was already accurate.
score: 5
**Input:**
```html
{block_html}
```
"""

    def handle_image_rotation(self, children: List[TableCell], image: Image.Image):
        ratios = [c.polygon.width / c.polygon.height for c in children]
        if len(ratios) < 2:
            return image

        is_rotated = all([r < self.rotation_max_wh_ratio for r in ratios])
        if not is_rotated:
            return image

        first_col_id = min([c.col_id for c in children])
        first_col = [c for c in children if c.col_id == first_col_id]
        first_col_cell = first_col[0]

        last_col_id = max([c.col_id for c in children])
        if last_col_id == first_col_id:
            return image

        last_col_cell = [c for c in children if c.col_id == last_col_id][0]
        cell_diff = first_col_cell.polygon.y_start - last_col_cell.polygon.y_start
        if cell_diff == 0:
            return image

        if cell_diff > 0:
            return image.rotate(270, expand=True)
        else:
            return image.rotate(90, expand=True)

    def process_rewriting(self, document: Document, page: PageGroup, block: Table):
        children: List[TableCell] = block.contained_blocks(
            document, (BlockTypes.TableCell,)
        )
        if not children:
            if block.html:
                # Full-mode table - the html came straight from table rec
                self.rewrite_html_table(document, page, block)
            return

        # LLMs don't handle tables with a lot of rows very well
        unique_rows = set([cell.row_id for cell in children])
        row_count = len(unique_rows)
        row_idxs = sorted(list(unique_rows))

        if row_count > self.max_table_rows:
            return

        # Inference by chunk to handle long tables better
        parsed_cells = []
        row_shift = 0
        block_image = self.extract_image(document, block)
        highres_size = page.get_image(highres=True).size
        block_rescaled_bbox = block.polygon.rescale(
            page.polygon.size, highres_size
        ).bbox
        for i in range(0, row_count, self.max_rows_per_batch):
            batch_row_idxs = row_idxs[i : i + self.max_rows_per_batch]
            batch_cells = [cell for cell in children if cell.row_id in batch_row_idxs]
            batch_cell_bboxes = [
                cell.polygon.rescale(page.polygon.size, highres_size).bbox
                for cell in batch_cells
            ]
            # bbox relative to the block
            batch_bbox = [
                min([bbox[0] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[0],
                min([bbox[1] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[1],
                max([bbox[2] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[0],
                max([bbox[3] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[1],
            ]
            if i == 0:
                # Ensure first image starts from the beginning
                batch_bbox[0] = 0
                batch_bbox[1] = 0
            elif i > row_count - self.max_rows_per_batch + 1:
                # Ensure final image grabs the entire height and width
                batch_bbox[2] = block_image.size[0]
                batch_bbox[3] = block_image.size[1]

            batch_image = block_image.crop(batch_bbox)
            block_html = block.format_cells(document, [], None, batch_cells)
            batch_image = self.handle_image_rotation(batch_cells, batch_image)
            batch_parsed_cells = self.rewrite_single_chunk(
                page, block, block_html, batch_cells, batch_image
            )
            if batch_parsed_cells is None:
                return  # Error occurred or no corrections needed

            batch_max_row = (
                max((cell.row_id for cell in batch_parsed_cells), default=-1) + 1
            )
            for cell in batch_parsed_cells:
                cell.row_id += row_shift
                parsed_cells.append(cell)
            row_shift += batch_max_row

        block.structure = []
        for cell in parsed_cells:
            page.add_full_block(cell)
            block.add_structure(cell)

    def rewrite_html_table(self, document: Document, page: PageGroup, block: Table):
        """Rewrite a table whose content is raw html from full-mode table rec
        (no TableCell children). The corrected html is written back to
        block.html."""
        row_count = block.html.count("<tr")
        if row_count > self.max_table_rows:
            return

        block_image = self.extract_image(document, block)
        block_html = block.html
        for _ in range(self.max_table_iterations):
            prompt = self.table_rewriting_prompt.replace("{block_html}", block_html)
            response = self.llm_service(prompt, block_image, block, TableSchema)

            if not response or "corrected_html" not in response:
                block.update_metadata(llm_error_count=1)
                return

            corrected_html = response["corrected_html"]
            if "no corrections needed" in corrected_html.lower():
                return

            corrected_html = (
                corrected_html.strip()
                .removeprefix("```html")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            if not corrected_html.endswith("</table>"):
                block.update_metadata(llm_error_count=1)
                return

            # Sanity check the html parses into enough cells
            if len(self.parse_html_table(corrected_html, block, page)) <= 1:
                block.update_metadata(llm_error_count=1)
                return

            block_html = corrected_html
            score = response.get("score", 5)
            if score >= 4:
                break

        block.html = block_html

    def rewrite_single_chunk(
        self,
        page: PageGroup,
        block: Block,
        block_html: str,
        children: List[TableCell],
        image: Image.Image,
        total_iterations: int = 0,
    ):
        prompt = self.table_rewriting_prompt.replace("{block_html}", block_html)

        response = self.llm_service(prompt, image, block, TableSchema)

        if not response or "corrected_html" not in response:
            block.update_metadata(llm_error_count=1)
            return

        corrected_html = response["corrected_html"]

        # The original table is okay
        if "no corrections needed" in corrected_html.lower():
            return

        corrected_html = (
            corrected_html.strip()
            .removeprefix("```html")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        # Re-iterate if low score
        total_iterations += 1
        score = response.get("score", 5)
        analysis = response.get("analysis", "")
        logger.debug(f"Got table rewriting score {score} with analysis: {analysis}")
        if total_iterations < self.max_table_iterations and score < 4:
            logger.info(
                f"Table rewriting low score {score}, on iteration {total_iterations}"
            )
            block_html = corrected_html
            return self.rewrite_single_chunk(
                page, block, block_html, children, image, total_iterations
            )

        parsed_cells = self.parse_html_table(corrected_html, block, page)
        if len(parsed_cells) <= 1:
            block.update_metadata(llm_error_count=1)
            logger.debug(f"Table parsing issue, only {len(parsed_cells)} cells found")
            return

        if not corrected_html.endswith("</table>"):
            logger.debug(
                "Table parsing issue, corrected html does not end with </table>"
            )
            block.update_metadata(llm_error_count=1)
            return

        return parsed_cells

    @staticmethod
    def get_cell_text(element, keep_tags=("br", "i", "b", "span", "math")) -> str:
        for tag in element.find_all(True):
            if tag.name not in keep_tags:
                tag.unwrap()
        return element.decode_contents()

    def parse_html_table(
        self, html_text: str, block: Block, page: PageGroup
    ) -> List[TableCell]:
        soup = BeautifulSoup(html_text, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        # Initialize grid
        rows = table.find_all("tr")
        cells = []

        # Find maximum number of columns in colspan-aware way
        max_cols = 0
        for row in rows:
            row_tds = row.find_all(["td", "th"])
            curr_cols = 0
            for cell in row_tds:
                colspan = int(cell.get("colspan", 1))
                curr_cols += colspan
            if curr_cols > max_cols:
                max_cols = curr_cols

        grid = [[True] * max_cols for _ in range(len(rows))]

        for i, row in enumerate(rows):
            cur_col = 0
            row_cells = row.find_all(["td", "th"])
            for j, cell in enumerate(row_cells):
                while cur_col < max_cols and not grid[i][cur_col]:
                    cur_col += 1

                if cur_col >= max_cols:
                    logger.info("Table parsing warning: too many columns found")
                    break

                cell_text = self.get_cell_text(cell).strip()
                rowspan = min(int(cell.get("rowspan", 1)), len(rows) - i)
                colspan = min(int(cell.get("colspan", 1)), max_cols - cur_col)
                cell_rows = list(range(i, i + rowspan))
                cell_cols = list(range(cur_col, cur_col + colspan))

                if colspan == 0 or rowspan == 0:
                    logger.info("Table parsing issue: invalid colspan or rowspan")
                    continue

                for r in cell_rows:
                    for c in cell_cols:
                        grid[r][c] = False

                cell_bbox = [
                    block.polygon.bbox[0] + cur_col,
                    block.polygon.bbox[1] + i,
                    block.polygon.bbox[0] + cur_col + colspan,
                    block.polygon.bbox[1] + i + rowspan,
                ]
                cell_polygon = PolygonBox.from_bbox(cell_bbox)

                cell_obj = TableCell(
                    text_lines=[cell_text],
                    row_id=i,
                    col_id=cur_col,
                    rowspan=rowspan,
                    colspan=colspan,
                    is_header=cell.name == "th",
                    polygon=cell_polygon,
                    page_id=page.page_id,
                )
                cells.append(cell_obj)
                cur_col += colspan

        return cells


class TableSchema(BaseModel):
    comparison: str
    corrected_html: str
    analysis: str
    score: int
