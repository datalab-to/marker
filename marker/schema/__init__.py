from enum import auto, Enum


class BlockTypes(str, Enum):
    Line = auto()
    Span = auto()
    Char = auto()
    FigureGroup = auto()
    TableGroup = auto()
    ListGroup = auto()
    PictureGroup = auto()
    Page = auto()
    Caption = auto()
    Code = auto()
    Figure = auto()
    Footnote = auto()
    Form = auto()
    Equation = auto()
    Handwriting = auto()
    TextInlineMath = auto()
    ListItem = auto()
    PageFooter = auto()
    PageHeader = auto()
    Picture = auto()
    SectionHeader = auto()
    Table = auto()
    Text = auto()
    TableOfContents = auto()
    Document = auto()
    ComplexRegion = auto()
    TableCell = auto()
    Reference = auto()
    Marginal = auto()

    def __str__(self):
        return self.name
