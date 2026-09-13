import pytest

from marker.builders.line import has_deterministic_garble
from marker.renderers.markdown import cleanup_text

# CPU-only: pure pattern checks, no fixtures or models.
pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    "text,expected",
    [
        # Lying no-ToUnicode font damage (glyph codes one ASCII band down)
        ("4HE QUICK 9OU", True),  # T -> glyph named "four", Y -> "nine"
        ("5NDER THE TABLE", True),  # U -> "five"
        ("7HAT DOES 7HAT MEAN", True),  # W -> "seven", twice
        ("CLARIlED THE lELD", True),  # fi/fl ligature glue
        ("ENVIRONMENTAL INmUENCES", True),  # fl ligature glue
        ("the decision-making of THE GOVERNMENTv", True),  # closing quote glyph
        ("IMAGINE BEING hWRONGv", True),  # quote glyph pair
        ("SEEING IT hANALYSIS AND", True),  # opening quote glyph
        ("CONSCIENCEx-UST THE CITIZEN", True),  # ellipsis glyph
        ("comma leaked as A\x0cB", True),  # punctuation codes as C0 bytes
        ("period A\x0eB question A\x1fB", True),
        ("delete byte A\x7fB", True),
        (
            "how to: s interpret events from multiple views. s find sources. "
            "s identify the viewpoints",
            True,
        ),  # flattened bullet list
        # Clean text must not trigger re-OCR
        ("WHAT IS THE MAIN IDEA?", False),
        ("7HAT IS", False),  # one short shifted token alone is not enough
        ("9OU HAVE", False),
        ("Reading Reflectively", False),
        ("1ST place and 22ND row in 2024", False),  # ordinals
        ("by 9AM sharp, back at 3PM", False),  # clock times
        ("over 4GB of downloads", False),
        ("the Sv-Tx and Ov rows, plus Lv.", False),  # single-letter codes
        ("ISBN 978-92-76-48882-8KJ-NA-EN", False),
        ("Siehe https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=%2FTXT", False),
        ("mit 1200 hDPI", False),
        ("nach 0,3 s wird noch 10 s gewartet", False),  # German seconds
        ("T in s (am Graphen abgelesen)", False),
        ("Reported (NASA) findings today.", False),
        ("AlJazeera and IMDb report", False),
        ("A\tB and C\nD", False),  # tab/newline are legitimate whitespace
        ("Plain prose with an em-dash — and ünïcödé.", False),
        ("s one. s two", False),  # two standalone "s" are not a bullet list
    ],
)
def test_deterministic_garble(text, expected):
    assert has_deterministic_garble(text) is expected


def test_cleanup_text_strips_control_bytes():
    assert cleanup_text("A\x0cB") == "AB"
    assert cleanup_text("q\x1f\nnext") == "q\nnext"
    assert cleanup_text("keep\ttab\nand\rreturn") == "keep\ttab\nand\rreturn"
    assert cleanup_text("line\n\n\n\nline") == "line\n\nline"
