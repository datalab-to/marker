import pytest

from marker.util import strip_code_fence


@pytest.mark.parametrize(
    "text,lang,expected",
    [
        # Fenced responses are unwrapped, with or without a language tag.
        ("```markdown\n# Title\n```", "markdown", "# Title"),
        ("```\n# Title\n```", "markdown", "# Title"),
        ("```html\n<table></table>\n```", "html", "<table></table>"),
        ("  ```html\n<table></table>\n```  ", "html", "<table></table>"),
        # Unfenced responses are returned untouched, even when they start with
        # characters that appear in the fence marker. `lstrip` would eat those.
        ("and the results were clear", "markdown", "and the results were clear"),
        ("a photo of a dog", "markdown", "a photo of a dog"),
        ("normally we do X", "markdown", "normally we do X"),
        ("markdown", "markdown", "markdown"),
        ("the table", "html", "the table"),
        ("html tags here", "html", "html tags here"),
        ("<table></table>", "html", "<table></table>"),
    ],
)
def test_strip_code_fence(text, lang, expected):
    assert strip_code_fence(text, lang) == expected
