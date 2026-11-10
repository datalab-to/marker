"""Mapping between surya's canonical layout labels and marker BlockTypes."""

from marker.schema import BlockTypes

# Surya canonical layout labels -> marker block types. BlankPage is handled
# separately (the box is dropped and the page left empty).
SURYA_LABEL_TO_BLOCK_TYPE = {
    "Text": BlockTypes.Text,
    "SectionHeader": BlockTypes.SectionHeader,
    "PageHeader": BlockTypes.PageHeader,
    "PageFooter": BlockTypes.PageFooter,
    "Caption": BlockTypes.Caption,
    "Footnote": BlockTypes.Footnote,
    "Code": BlockTypes.Code,
    "Bibliography": BlockTypes.Bibliography,
    "Picture": BlockTypes.Picture,
    "Figure": BlockTypes.Figure,
    "Diagram": BlockTypes.Diagram,
    "Table": BlockTypes.Table,
    "Form": BlockTypes.Form,
    "Equation": BlockTypes.Equation,
    "Formula": BlockTypes.Equation,
    "ListGroup": BlockTypes.ListGroup,
    "TableOfContents": BlockTypes.TableOfContents,
    "ChemicalBlock": BlockTypes.ChemicalBlock,
}

BLANK_PAGE_LABEL = "BlankPage"

# Inverse mapping for building synthetic surya LayoutResults from marker
# blocks (block-mode OCR). Marker-only types fall back to "Text".
BLOCK_TYPE_TO_SURYA_LABEL = {v: k for k, v in SURYA_LABEL_TO_BLOCK_TYPE.items()}
BLOCK_TYPE_TO_SURYA_LABEL.update(
    {
        BlockTypes.TextInlineMath: "Text",
        BlockTypes.Handwriting: "Text",
        BlockTypes.ListItem: "Text",
        BlockTypes.ComplexRegion: "Figure",
    }
)


def surya_label_to_block_type(label: str) -> BlockTypes | None:
    """None means the box should be dropped (unknown or BlankPage)."""
    return SURYA_LABEL_TO_BLOCK_TYPE.get(label)


def block_type_to_surya_label(block_type: BlockTypes) -> str:
    return BLOCK_TYPE_TO_SURYA_LABEL.get(block_type, "Text")
