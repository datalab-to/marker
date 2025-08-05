import re
from collections import defaultdict
from typing import Annotated, Tuple

import regex
from bs4 import NavigableString
from markdownify import MarkdownConverter
from pydantic import BaseModel

from marker.renderers.html import HTMLRenderer
from marker.schema import BlockTypes
from marker.schema.document import Document


def escape_dollars(text):
    return text.replace("$", r"\$")

def cleanup_text(full_text):
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r'(\n\s){3,}', '\n\n', full_text)
    return full_text.strip()

def get_formatted_table_text(element):

    text = []
    for content in element.contents:
        if content is None:
            continue

        if isinstance(content, NavigableString):
            stripped = content.strip()
            if stripped:
                text.append(escape_dollars(stripped))
        elif content.name == 'br':
            text.append('<br>')
        elif content.name == "math":
            text.append("$" + content.text + "$")
        else:
            content_str = escape_dollars(str(content))
            text.append(content_str)

    full_text = ""
    for i, t in enumerate(text):
        if t == '<br>':
            full_text += t
        elif i > 0 and text[i - 1] != '<br>':
            full_text += " " + t
        else:
            full_text += t
    return full_text


class Markdownify(MarkdownConverter):
    def __init__(self, paginate_output, page_separator, inline_math_delimiters, block_math_delimiters, **kwargs):
        super().__init__(**kwargs)
        self.paginate_output = paginate_output
        self.page_separator = page_separator
        self.inline_math_delimiters = inline_math_delimiters
        self.block_math_delimiters = block_math_delimiters
        self.image_mappings = {}  # Store img_id -> url mappings

    def convert_div(self, el, text, convert_as_inline):
        is_page = el.has_attr('class') and el['class'][0] == 'page'
        if self.paginate_output and is_page:
            page_id = el['data-page-id']
            pagination_item = "\n\n" + "{" + str(page_id) + "}" + self.page_separator + "\n\n"
            return pagination_item + text
        else:
            return text

    def convert_p(self, el, text, convert_as_inline):
        hyphens = r'-—¬'
        has_continuation = el.has_attr('class') and 'has-continuation' in el['class']
        if has_continuation:
            block_type = BlockTypes[el['block-type']]
            if block_type in [BlockTypes.TextInlineMath, BlockTypes.Text]:
                if regex.compile(rf'.*[\p{{Ll}}|\d][{hyphens}]\s?$', regex.DOTALL).match(text):  # handle hypenation across pages
                    return regex.split(rf"[{hyphens}]\s?$", text)[0]
                return f"{text} "
            if block_type == BlockTypes.ListGroup:
                return f"{text}"
        return f"{text}\n\n" if text else ""  # default convert_p behavior

    def convert_math(self, el, text, convert_as_inline):
        block = (el.has_attr('display') and el['display'] == 'block')
        if block:
            return "\n" + self.block_math_delimiters[0] + text + self.block_math_delimiters[1] + "\n"
        else:
            return " " + self.inline_math_delimiters[0] + text + self.inline_math_delimiters[1] + " "


    def convert_table(self, el, text, convert_as_inline):
        total_rows = len(el.find_all('tr'))
        colspans = []
        rowspan_cols = defaultdict(int)
        for i, row in enumerate(el.find_all('tr')):
            row_cols = rowspan_cols[i]
            for cell in row.find_all(['td', 'th']):
                colspan = int(cell.get('colspan', 1))
                row_cols += colspan
                for r in range(int(cell.get('rowspan', 1)) - 1):
                    rowspan_cols[i + r] += colspan # Add the colspan to the next rows, so they get the correct number of columns
            colspans.append(row_cols)
        total_cols = max(colspans) if colspans else 0

        grid = [[None for _ in range(total_cols)] for _ in range(total_rows)]

        for row_idx, tr in enumerate(el.find_all('tr')):
            col_idx = 0
            for cell in tr.find_all(['td', 'th']):
                # Skip filled positions
                while col_idx < total_cols and grid[row_idx][col_idx] is not None:
                    col_idx += 1

                # Fill in grid
                value = get_formatted_table_text(cell).replace("\n", " ").replace("|", " ").strip()
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))

                if col_idx >= total_cols:
                    # Skip this cell if we're out of bounds
                    continue

                for r in range(rowspan):
                    for c in range(colspan):
                        try:
                            if r == 0 and c == 0:
                                grid[row_idx][col_idx] = value
                            else:
                                grid[row_idx + r][col_idx + c] = '' # Empty cell due to rowspan/colspan
                        except IndexError:
                            # Sometimes the colspan/rowspan predictions can overflow
                            print(f"Overflow in columns: {col_idx + c} >= {total_cols} or rows: {row_idx + r} >= {total_rows}")
                            continue

                col_idx += colspan

        markdown_lines = []
        col_widths = [0] * total_cols
        for row in grid:
            for col_idx, cell in enumerate(row):
                if cell is not None:
                    col_widths[col_idx] = max(col_widths[col_idx], len(str(cell)))

        add_header_line = lambda: markdown_lines.append('|' + '|'.join('-' * (width + 2) for width in col_widths) + '|')

        # Generate markdown rows
        added_header = False
        for i, row in enumerate(grid):
            is_empty_line = all(not cell for cell in row)
            if is_empty_line and not added_header:
                # Skip leading blank lines
                continue

            line = []
            for col_idx, cell in enumerate(row):
                if cell is None:
                    cell = ''
                padding = col_widths[col_idx] - len(str(cell))
                line.append(f" {cell}{' ' * padding} ")
            markdown_lines.append('|' + '|'.join(line) + '|')

            if not added_header:
                # Skip empty lines when adding the header row
                add_header_line()
                added_header = True

        # Handle one row tables
        if total_rows == 1:
            add_header_line()

        table_md = '\n'.join(markdown_lines)
        return "\n\n" + table_md + "\n\n"

    def convert_a(self, el, text, convert_as_inline):
        text = self.escape(text)
        # Escape brackets and parentheses in text
        text = re.sub(r"([\[\]()])", r"\\\1", text)
        return super().convert_a(el, text, convert_as_inline)

    def convert_span(self, el, text, convert_as_inline):
        if el.get("id"):
            return f'<span id="{el["id"]}">{text}</span>'
        else:
            return text

    def escape(self, text):
        text = super().escape(text)
        if self.options['escape_dollars']:
            text = text.replace('$', r'\$')
        return text

class MarkdownOutput(BaseModel):
    markdown: str
    images: dict
    mol_images: dict
    table_contents: dict
    metadata: dict


class MarkdownRenderer(HTMLRenderer):
    page_separator: Annotated[str, "The separator to use between pages.", "Default is '-' * 48."] = "-" * 48
    inline_math_delimiters: Annotated[Tuple[str], "The delimiters to use for inline math."] = ("$", "$")
    block_math_delimiters: Annotated[Tuple[str], "The delimiters to use for block math."] = ("$$", "$$")

    @property
    def md_cls(self):
        return Markdownify(
            self.paginate_output,
            self.page_separator,
            heading_style="ATX",
            bullets="-",
            escape_misc=False,
            escape_underscores=True,
            escape_asterisks=True,
            escape_dollars=True,
            sub_symbol="<sub>",
            sup_symbol="<sup>",
            inline_math_delimiters=self.inline_math_delimiters,
            block_math_delimiters=self.block_math_delimiters
        )

    def process_images_for_markdown(self, images):
        """
        Process images dict to handle S3 URLs and local images appropriately for Markdown
        
        Args:
            images: Dict containing image data (either URLs or binary data)
            
        Returns:
            Processed images dict suitable for Markdown output
        """
        processed_images = {}
        picture_images = {}
        molecule_img_images = {}
        molecule_table_images = {}
        
        for key, value in images.items():
            if isinstance(value, dict) and value.get("extra_type", "") == "molecule_img":
                # For S3 images, we store the URL info but don't include binary data
                molecule_img_images[key] = {
                    "url": value["url"],
                    "type": "s3",
                    "key": value.get("key", ""),
                    "extra_type": value.get("extra_type", ""),
                    "smiles": value.get("smiles", ""),
                    "mol_block": value.get("mol_block", ""),
                    "label": value.get("label", ""),
                    "page_idx": value.get("page_idx", ""),
                    "bbox": value.get("bbox", []),
                    "original_name": value.get("original_name", str(key))
                }
            elif isinstance(value, dict) and value.get("extra_type", "") == "molecule_table":
                molecule_table_images[key] = {
                    "url": value["url"],
                    "type": "s3",
                    "key": value.get("key", ""),
                    "extra_type": value.get("extra_type", ""),
                    "html_content": value.get("html_content", ""),
                    "page_idx": value.get("page_idx", ""),
                    "bbox": value.get("bbox", ""),
                    "original_name": value.get("original_name", str(key))
                }
            elif isinstance(value, dict) and value.get("extra_type", "") == "picture":
                picture_images[key] = {
                    "url": value["url"],
                    "type": "s3",
                    "key": value.get("key", ""),
                    "extra_type": value.get("extra_type", ""),
                    "original_name": value.get("original_name", str(key))
                }
            else:
                # For local/base64 images, keep as is
                processed_images[key] = value
                
        return processed_images, picture_images, molecule_img_images, molecule_table_images

    def __call__(self, document: Document) -> MarkdownOutput:
        document_output = document.render()
        full_html, images = self.extract_html(document, document_output)
        
        # Extract image mappings and table contents from images dict
        # Process images for Markdown
        _, picture_images, molecule_img_images, molecule_table_images = self.process_images_for_markdown(images)
        print("@@@@ picture_images: ", picture_images)
        print("@@@@ molecule_img_images: ", molecule_img_images)
        print("@@@@ molecule_table_images: ", molecule_table_images)
        markdown = self.md_cls.convert(full_html)
        markdown = cleanup_text(markdown)
        
        return MarkdownOutput(
            markdown=markdown,
            images=picture_images,
            mol_images=molecule_img_images,
            table_contents=molecule_table_images,
            metadata=self.generate_document_metadata(document, document_output)
        )
