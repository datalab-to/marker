"""Characters a PDF hands over that belong to the layout rather than a word.

PDFium reports what the font's `/ToUnicode` map claims, and `ftfy.fix_text`
leaves all of this alone: it repairs mojibake, not a font telling the truth
about a glyph nobody can render.
"""

# Adobe's Corporate Use Subarea, limited to the glyphs Adobe's own list names as
# variants of a character that has a Unicode value.  A book set in oldstyle
# figures otherwise reports its dates as unrenderable private-use codes.
ADOBE_GLYPH_VARIANTS = {
    0xF6D9: 0x00A9,  # copyrightserif
    0xF6DB: 0x2122,  # trademarkserif
    0xF724: 0x0024,  # dollaroldstyle
    0xF730: 0x0030,  # zerooldstyle
    0xF731: 0x0031,  # oneoldstyle
    0xF732: 0x0032,  # twooldstyle
    0xF733: 0x0033,  # threeoldstyle
    0xF734: 0x0034,  # fouroldstyle
    0xF735: 0x0035,  # fiveoldstyle
    0xF736: 0x0036,  # sixoldstyle
    0xF737: 0x0037,  # sevenoldstyle
    0xF738: 0x0038,  # eightoldstyle
    0xF739: 0x0039,  # nineoldstyle
    0xF7A2: 0x00A2,  # centoldstyle
    0xF8E9: 0x00A9,  # copyrightsans
    0xF8EA: 0x2122,  # trademarksans
}

# Invisible characters that describe how a word may be broken, not what it says.
# The zero-width joiners are deliberately absent: they carry meaning in Indic,
# Arabic and Persian text and in emoji sequences, so dropping them would corrupt
# real content.
DISCARDABLE_FORMATTING = {
    "­",  # soft hyphen
    "﻿",  # byte order mark / zero width no-break space
}


def resolve_adobe_glyph_variants(text: str) -> str:
    """Replace Adobe Corporate Use Subarea glyphs with the characters they name.

    Private-use characters from anywhere else are left exactly as extracted.
    A font's assignment means nothing outside that font, and a producer that
    maps part of a subset to real characters and leaves the rest private is
    stating its own limit -- resolving those would be inventing text.
    """
    if not text:
        return text

    return "".join(
        chr(ADOBE_GLYPH_VARIANTS[ord(character)])
        if ord(character) in ADOBE_GLYPH_VARIANTS
        else character
        for character in text
    )


def is_discardable_formatting(character: str) -> bool:
    """True for a character that carries layout, never a word.

    Besides the invisible formatting above, this covers the Unicode
    *noncharacters* -- U+FDD0..U+FDEF and U+nFFFE/U+nFFFF in every plane.  They
    are permanently reserved and never valid in interchange, and PDFium hands
    them over where a PDF's font maps a hyphenation point to an unassigned slot.
    Left in the text they corrupt the word around them -- `de<U+FFFE>picted` --
    and hand a strict downstream consumer a document it must reject.
    """
    if character in DISCARDABLE_FORMATTING:
        return True

    code = ord(character)
    if 0xFDD0 <= code <= 0xFDEF:
        return True
    return code & 0xFFFE == 0xFFFE


def clean_extracted_text(text: str) -> str:
    """Resolve what the font meant, and drop what belongs to the layout.

    A discardable character immediately before a line break is a hyphenation
    point, exactly as a trailing "-" is, and becomes one: dropping it outright
    would leave a broken word with nothing to say it was broken, and marker
    would rejoin it as two words.  Everywhere else it simply goes.
    """
    if not text:
        return text

    resolved = resolve_adobe_glyph_variants(text)
    cleaned = []
    for position, character in enumerate(resolved):
        if not is_discardable_formatting(character):
            cleaned.append(character)
            continue

        following = resolved[position + 1 : position + 2]
        breaks_line = following in ("\n", "\r")
        if breaks_line and cleaned[-1:] != ["-"]:
            cleaned.append("-")

    return "".join(cleaned)
