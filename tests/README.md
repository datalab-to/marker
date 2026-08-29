# Running the tests

```bash
uv run pytest -m "not integration" -q
```

`integration`-marked tests run the VLM over real pages and are slow; they are
excluded above.  The `cpu` marker selects the tests that need no GPU or
inference server (`-m cpu`).

## Sample documents

Most tests convert a sample document named by a `filename` marker:

```python
@pytest.mark.filename("thinkpython.pdf")
def test_something(pdf_document): ...
```

Tests without the marker get `adversarial.pdf`.

Those documents come from the [`datalab-to/pdfs`][dataset] dataset on the
HuggingFace Hub, which is **gated** — the Hub answers HTTP 401 without an
access token.  When a document cannot be found, the test **skips with a reason
naming the file and both sources** rather than erroring.  A full run then looks
like:

```
65 passed, 73 skipped, 3 deselected
```

`tests/dataset.py` resolves each document from two sources, in order.

### 1. A local directory

Drop documents into `tests/data/pdfs/`, named exactly as the tests request
them (`adversarial.pdf`, `gatsby.docx`, `single_sheet.xlsx`, ...).  Point
somewhere else with:

```bash
MARKER_TEST_PDF_DIR=/path/to/samples uv run pytest -m "not integration" -q
```

A local file wins over the Hub, so a fully populated directory runs the suite
with no Hub access at all.

Use the **real** documents: the tests assert on their actual content (for
example `"# Subspace Adversarial Training"` from `adversarial.pdf`, and a
12-page length).  A stand-in document loads fine but fails those assertions.

Documents the suite currently asks for:

| Type | Documents |
| --- | --- |
| PDF | `A17_FlightPlan.pdf`, `adversarial.pdf`, `adversarial_rot.pdf`, `arxiv_test.pdf`, `bio_pdf.pdf`, `form_1040.pdf`, `handwritten.pdf`, `hindi_judgement.pdf`, `multicol-blocks.pdf`, `population_stats.pdf`, `pres.pdf`, `table_ex.pdf`, `table_ex2.pdf`, `thinkpython.pdf`, `water_damage.pdf` |
| Other | `china.html`, `gatsby.docx`, `lambda.pptx`, `manual.epub`, `single_sheet.xlsx` |

Regenerate that list with:

```bash
grep -rho 'pytest.mark.filename("[^"]*")' tests/ | sed 's/.*("//;s/")//' | sort -u
```

### 2. The HuggingFace Hub

Request access at <https://huggingface.co/datasets/datalab-to/pdfs>, then
authenticate once:

```bash
uv run huggingface-cli login
```

Or set a token in the environment:

```bash
export HF_TOKEN=hf_...
```

Verify access with:

```bash
uv run python -c "import datasets; print(datasets.load_dataset('datalab-to/pdfs', split='train'))"
```

[dataset]: https://huggingface.co/datasets/datalab-to/pdfs
