from typing import Dict, Type
from importlib import import_module

from marker.schema import BlockTypes
from marker.schema.blocks import (
    Block,
    Caption,
    Code,
    Equation,
    Figure,
    Footnote,
    Form,
    Handwriting,
    InlineMath,
    ListItem,
    PageFooter,
    PageHeader,
    Picture,
    SectionHeader,
    Table,
    TableOfContents,
    Text,
    ComplexRegion,
    TableCell,
    Reference,
    Marginal,
)
from marker.schema.document import Document
from marker.schema.groups import (
    FigureGroup,
    ListGroup,
    PageGroup,
    PictureGroup,
    TableGroup,
)
from marker.schema.text import Line, Span
from marker.schema.text.char import Char

BLOCK_REGISTRY: Dict[BlockTypes, str] = {}


def register_block_class(block_type: BlockTypes, block_cls: Type[Block]):
    BLOCK_REGISTRY[block_type] = f"{block_cls.__module__}.{block_cls.__name__}"


def get_block_class(block_type: BlockTypes) -> Type[Block]:
    class_path = BLOCK_REGISTRY[block_type]
    module_name, class_name = class_path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


register_block_class(BlockTypes.Line, Line)
register_block_class(BlockTypes.Span, Span)
register_block_class(BlockTypes.Char, Char)
register_block_class(BlockTypes.FigureGroup, FigureGroup)
register_block_class(BlockTypes.TableGroup, TableGroup)
register_block_class(BlockTypes.ListGroup, ListGroup)
register_block_class(BlockTypes.PictureGroup, PictureGroup)
register_block_class(BlockTypes.Page, PageGroup)
register_block_class(BlockTypes.Caption, Caption)
register_block_class(BlockTypes.Code, Code)
register_block_class(BlockTypes.Figure, Figure)
register_block_class(BlockTypes.Footnote, Footnote)
register_block_class(BlockTypes.Form, Form)
register_block_class(BlockTypes.Equation, Equation)
register_block_class(BlockTypes.Handwriting, Handwriting)
register_block_class(BlockTypes.TextInlineMath, InlineMath)
register_block_class(BlockTypes.ListItem, ListItem)
register_block_class(BlockTypes.PageFooter, PageFooter)
register_block_class(BlockTypes.PageHeader, PageHeader)
register_block_class(BlockTypes.Picture, Picture)
register_block_class(BlockTypes.SectionHeader, SectionHeader)
register_block_class(BlockTypes.Table, Table)
register_block_class(BlockTypes.Text, Text)
register_block_class(BlockTypes.TableOfContents, TableOfContents)
register_block_class(BlockTypes.ComplexRegion, ComplexRegion)
register_block_class(BlockTypes.TableCell, TableCell)
register_block_class(BlockTypes.Reference, Reference)
register_block_class(BlockTypes.Marginal, Marginal)
register_block_class(BlockTypes.Document, Document)

assert len(BLOCK_REGISTRY) == len(BlockTypes)
assert all(
    [
        get_block_class(k).model_fields["block_type"].default == k
        for k, _ in BLOCK_REGISTRY.items()
    ]
)
