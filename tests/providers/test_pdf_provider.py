import pytest
from marker.providers.pdf import PdfProvider


@pytest.mark.config({"page_range": [0]})
def test_pdf_provider(doc_provider):
    assert len(doc_provider) == 12
    assert doc_provider.get_images([0], 72)[0].size == (612, 792)
    assert doc_provider.get_images([0], 96)[0].size == (816, 1056)

    page_lines = doc_provider.get_page_lines(0)
    assert len(page_lines) == 85

    spans = page_lines[0].spans
    assert len(spans) == 2
    assert spans[0].text == "Subspace Adversarial Training"
    assert spans[0].font == "NimbusRomNo9L-Medi"
    assert spans[0].formats == ["plain"]


class TestFontFlagsDetection:
    """Unit tests for font flag-based format detection."""

    @pytest.fixture
    def provider(self):
        provider = object.__new__(PdfProvider)
        return provider

    def test_italic_flag_only(self, provider):
        flags = 1 << 6
        result = provider.font_flags_to_format(flags)
        assert "italic" in result
        assert "bold" not in result
        assert "plain" not in result

    def test_bold_flag_only(self, provider):
        flags = 1 << 18
        result = provider.font_flags_to_format(flags)
        assert "bold" in result
        assert "italic" not in result
        assert "plain" not in result

    def test_bold_and_italic_flags_together(self, provider):
        flags = (1 << 6) | (1 << 18)
        result = provider.font_flags_to_format(flags)
        assert "italic" in result
        assert "bold" in result
        assert "plain" not in result

    def test_symbolic_and_italic_preserved(self, provider):
        flags = (1 << 2) | (1 << 6)
        result = provider.font_flags_to_format(flags)
        assert "italic" in result, "Symbolic fonts should not strip italic formatting"
        assert "plain" not in result

    def test_symbolic_only_is_plain(self, provider):
        flags = 1 << 2
        result = provider.font_flags_to_format(flags)
        assert "plain" in result
        assert "italic" not in result
        assert "bold" not in result

    def test_no_flags_is_plain(self, provider):
        flags = 0
        result = provider.font_flags_to_format(flags)
        assert "plain" in result

    def test_none_flags_is_plain(self, provider):
        result = provider.font_flags_to_format(None)
        assert result == {"plain"}

    def test_use_extern_attr_only_is_plain(self, provider):
        flags = 1 << 19
        result = provider.font_flags_to_format(flags)
        assert "plain" in result

    def test_complex_flag_combination_with_italic(self, provider):
        flags = (1 << 1) | (1 << 2) | (1 << 6) | (1 << 0)
        result = provider.font_flags_to_format(flags)
        assert "italic" in result
        assert "plain" not in result


class TestFontNameDetection:
    """Unit tests for font name-based format detection."""

    @pytest.fixture
    def provider(self):
        provider = object.__new__(PdfProvider)
        return provider

    def test_italic_standard(self, provider):
        result = provider.font_names_to_format("Arial-Italic")
        assert "italic" in result
        assert "bold" not in result

    def test_italic_oblique(self, provider):
        result = provider.font_names_to_format("Helvetica-Oblique")
        assert "italic" in result

    def test_italic_slant(self, provider):
        result = provider.font_names_to_format("CustomFont-Slanted")
        assert "italic" in result

    def test_italic_abbreviation_it(self, provider):
        result = provider.font_names_to_format("TimesIt")
        assert "italic" in result

    def test_italic_foreign_language_cursiva(self, provider):
        result = provider.font_names_to_format("FuenteCursiva")
        assert "italic" in result

    def test_italic_foreign_language_corsivo(self, provider):
        result = provider.font_names_to_format("FontCorsivo")
        assert "italic" in result

    def test_italic_foreign_language_kursiv(self, provider):
        result = provider.font_names_to_format("SchriftKursiv")
        assert "italic" in result

    def test_bold_standard(self, provider):
        result = provider.font_names_to_format("Arial-Bold")
        assert "bold" in result
        assert "italic" not in result

    def test_bold_abbreviation_bd(self, provider):
        result = provider.font_names_to_format("ArialBd")
        assert "bold" in result

    def test_bold_black_variant(self, provider):
        result = provider.font_names_to_format("Helvetica-Black")
        assert "bold" in result

    def test_bold_heavy_variant(self, provider):
        result = provider.font_names_to_format("Font-Heavy")
        assert "bold" in result

    def test_bold_and_italic_combined(self, provider):
        result = provider.font_names_to_format("Arial-BoldItalic")
        assert "bold" in result
        assert "italic" in result

    def test_case_insensitive_detection(self, provider):
        assert "italic" in provider.font_names_to_format("ARIAL-ITALIC")
        assert "italic" in provider.font_names_to_format("arial-italic")
        assert "italic" in provider.font_names_to_format("Arial-ItAlIc")
        assert "bold" in provider.font_names_to_format("ARIAL-BOLD")
        assert "bold" in provider.font_names_to_format("arial-bold")

    def test_none_font_name(self, provider):
        result = provider.font_names_to_format(None)
        assert result == set()

    def test_plain_font_name(self, provider):
        result = provider.font_names_to_format("Arial-Regular")
        assert result == set()

    def test_fontspring_savoy_italic(self, provider):
        test_cases = [
            "Savoy-Italic",
            "Savoy Italic",
            "SavoyIt",
            "FontspringSavoy-Italic",
        ]
        for font_name in test_cases:
            result = provider.font_names_to_format(font_name)
            assert "italic" in result, f"Should detect italic in '{font_name}'"
