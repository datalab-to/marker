from types import SimpleNamespace

from marker.schema import BlockTypes
from marker.schema.blocks.listitem import replace_bullets


def test_replace_bullets_keeps_inner_dashes():
    # Regression for #1024: removing the bullet must not drop a dash inside the text.
    line = SimpleNamespace(
        children=[],
        id=SimpleNamespace(block_type=BlockTypes.Line),
        html="• He paused — then left.",
    )

    replace_bullets([line])

    assert "•" not in line.html
    assert line.html.strip() == "He paused — then left."
