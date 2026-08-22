"""Characters PDFium hands over that are not part of any word.

Written with explicit escapes throughout: every character under test is either
invisible or a private-use glyph, and a literal would be unreadable here and
easy to mangle in a later edit.
"""

from marker.providers.text_cleanup import (
    clean_extracted_text,
    is_discardable_formatting,
    resolve_adobe_glyph_variants,
)

OLDSTYLE_2019 = ""


def test_oldstyle_figures_become_digits():
    # A PDF setting "2019" in oldstyle figures hands over these glyph codes.
    assert resolve_adobe_glyph_variants(OLDSTYLE_2019) == "2019"


def test_symbol_variants_become_their_characters():
    assert resolve_adobe_glyph_variants(" ") == "© ©"  # serif, sans
    assert resolve_adobe_glyph_variants(" ") == "™ ™"
    assert resolve_adobe_glyph_variants("") == "$¢"


def test_private_use_outside_the_subarea_is_left_alone():
    # A font's own assignment means nothing outside that font, so resolving it
    # would be inventing text.  U+F8FF is Apple's logo, not a copyright sign.
    for character in ["", "", "\U000f0000", "\U00100000"]:
        assert resolve_adobe_glyph_variants(character) == character


def test_resolving_leaves_ordinary_text_untouched():
    assert resolve_adobe_glyph_variants("Ordinary text, 2019. ©") == (
        "Ordinary text, 2019. ©"
    )
    assert resolve_adobe_glyph_variants("") == ""


def test_noncharacters_are_discardable():
    for character in ["￾", "￿", "﷐", "﷯"]:
        assert is_discardable_formatting(character)
    # Reserved in every plane, not only the BMP.
    for character in ["\U0001fffe", "\U0010ffff"]:
        assert is_discardable_formatting(character)


def test_invisible_formatting_is_discardable():
    assert is_discardable_formatting("­")  # soft hyphen
    assert is_discardable_formatting("﻿")  # byte order mark / ZWNBSP


def test_zero_width_joiners_are_kept():
    # Marker reads Indic, Arabic and emoji text, where these carry meaning.
    for character in ["‌", "‍"]:
        assert not is_discardable_formatting(character)


def test_ordinary_characters_are_not_discardable():
    for character in ["a", " ", "\n", "-", "©", "�", "क"]:
        assert not is_discardable_formatting(character)


def test_a_soft_hyphen_inside_a_word_is_dropped():
    assert clean_extracted_text("hyphen­ation") == "hyphenation"


def test_a_noncharacter_inside_a_word_is_dropped():
    # PDFium hands these over where a font maps a hyphenation point to an
    # unassigned slot; left in place they corrupt the word around them.
    assert clean_extracted_text("de￾picted") == "depicted"


def test_resolving_and_discarding_happen_together():
    assert clean_extracted_text(f" {OLDSTYLE_2019}﻿") == "© 2019"


def test_an_emoji_joiner_sequence_survives():
    developer = "\U0001f469‍\U0001f4bb"
    assert clean_extracted_text(developer) == developer


def test_nothing_discardable_survives_a_clean():
    messy = f"a­b￾c﷐d﻿{OLDSTYLE_2019}"

    cleaned = clean_extracted_text(messy)

    assert cleaned == "abcd2019"
    assert not any(is_discardable_formatting(character) for character in cleaned)


def test_empty_input_is_returned_unchanged():
    assert clean_extracted_text("") == ""


def test_a_line_final_soft_hyphen_becomes_a_hyphenation_point():
    # Span text arrives as "hyphen<shy>\n".  Dropping the soft hyphen outright
    # would leave a line ending in no hyphen at all, and marker would rejoin it
    # as "hyphen ation"; a real hyphen is the signal the rest of marker reads.
    assert clean_extracted_text("hyphen­\n") == "hyphen-\n"


def test_a_line_final_noncharacter_becomes_a_hyphenation_point():
    assert clean_extracted_text("de￾\n") == "de-\n"


def test_a_mid_line_soft_hyphen_is_still_dropped():
    assert clean_extracted_text("co­operate") == "cooperate"


def test_a_hyphenation_point_is_not_doubled():
    assert clean_extracted_text("hyphen-­\n") == "hyphen-\n"


def test_a_trailing_soft_hyphen_with_no_line_break_is_dropped():
    # Without a following break there is nothing to rejoin, so inventing a
    # hyphen here would put one in the middle of a word.
    assert clean_extracted_text("hyphen­") == "hyphen"
