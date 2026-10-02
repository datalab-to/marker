import re
from bs4 import BeautifulSoup
from marker.schema import BlockTypes
from marker.schema.blocks import Block

try:
    import wordninja
    _WORDNINJA_LM = wordninja.DEFAULT_LANGUAGE_MODEL
except ImportError:
    _WORDNINJA_LM = None


def clean_caption_text(text: str) -> str:
    # 1. Punctuation and casing spacing:
    # Lowercase followed by uppercase (camelCase split)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Space after colon if followed by a letter (avoid http:// or numbers)
    text = re.sub(r"(:)([a-zA-Z])", r"\1 \2", text)
    # Space before opening parenthesis if preceded by a letter or digit
    text = re.sub(r"([a-zA-Z0-9])(\()", r"\1 \2", text)
    # Space after closing parenthesis if followed by a letter or digit
    text = re.sub(r"(\))([a-zA-Z0-9])", r"\1 \2", text)
    # Space after comma or semicolon if followed by a letter
    text = re.sub(r"([,;])([a-zA-Z])", r"\1 \2", text)

    # 2. Split agglutinated words within alphabetical tokens
    if _WORDNINJA_LM is not None:
        def replace_token(match):
            token = match.group(0)
            if len(token) <= 3 or token.isupper() or any(c.isdigit() for c in token):
                return token
            if token.lower() in _WORDNINJA_LM._wordcost:
                return token
            splits = wordninja.split(token)
            if len(splits) > 1 and all(
                s.lower() in _WORDNINJA_LM._wordcost or s in ("a", "A", "I") for s in splits
            ):
                return " ".join(splits)
            return token

        text = re.sub(r"[a-zA-Z]+", replace_token, text)
    return text


def clean_caption_html(html: str) -> str:
    if not html:
        return html
    soup = BeautifulSoup(html, "html.parser")
    for text_node in list(soup.find_all(string=True)):
        cleaned = clean_caption_text(str(text_node))
        text_node.replace_with(cleaned)
    return str(soup)


class Caption(Block):
    block_type: BlockTypes = BlockTypes.Caption
    block_description: str = "A text caption that is directly above or below an image or table. Only used for text describing the image or table.  "
    replace_output_newlines: bool = True
    html: str | None = None

    def assemble_html(self, document, child_blocks, parent_structure, block_config):
        if self.html:
            html = super().handle_html_output(
                document, child_blocks, parent_structure, block_config
            )
            html = clean_caption_html(html).strip()
            if html and not html.startswith("<p") and not html.startswith("<div"):
                html = f"<p>{html}</p>"
            return html

        template = super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
        return clean_caption_html(template)

