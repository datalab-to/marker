# Prefix-free list itemization — fork of datalab/marker

This fork removes marker's reliance on prefix-pattern heuristics (`a.`, `1.`,
`•`, etc.) for list detection. List items are surfaced as individual blocks
with their own bounding boxes, driven only by surya's layout model and
line-level layout. Useful when prefix-based detection produces false
positives on dense forms / problem sets.

Upstream: <https://github.com/datalab-to/marker>

## What changed

1. **`marker/builders/structure.py`** — disabled `group_lists` and
   `unmark_lists` in `StructureBuilder.__call__`. Surya-labeled `ListItem`
   blocks are no longer merged into `ListGroup`s, and isolated items are no
   longer demoted to `Text`. Each item stays as its own page-level block
   with its original bbox.

2. **`marker/processors/list_line_explode.py`** *(new)* — splits every
   multi-line `ListItem` into one `ListItem` per `Line` child. Each new
   item's polygon is the line's polygon. Prefix-free.

3. **`marker/processors/list_gap_cluster.py`** *(new)* — alternate
   strategy. Same input, but clusters lines by vertical gap: lines whose
   inter-line gap is at most `gap_factor * median_gap` (default 1.5×) stay
   merged into one `ListItem`; larger gaps start a new item. Preserves
   wrapped items in docs with even line spacing; collapses to a single item
   when there's no measurable gap between sub-items.

4. **`marker/converters/pdf.py`** — registers
   `ListItemLineExplodeProcessor` in the default pipeline immediately after
   `LineMergeProcessor`. To switch strategies, replace it with
   `ListItemGapClusterProcessor` in the `default_processors` tuple.

5. **`scripts/patch_surya_label.py`** *(new)* — patches the installed surya
   package so the layout model never emits the `<form>` → `Form` label;
   instead, those regions come through as `ListItem` so the patched
   structure builder and list-itemization processors handle them.
   Idempotent. Run once after installing or upgrading `surya-ocr`.

6. **`scripts/marker_view.py`** *(new)* — standalone HTML viewer. Renders
   each PDF page and overlays marker's JSON bboxes with color-coded labels
   and a "ListItems only" filter.

## Setup

```bash
pip install -e .
pip install surya-ocr pypdfium2
python scripts/patch_surya_label.py
```

## Usage

```bash
# convert
SURYA_INFERENCE_BACKEND=llamacpp marker_single path/to.pdf \
  --output_dir out --output_format json --disable_multiprocessing

# visualize
python scripts/marker_view.py path/to.pdf out/<stem>/<stem>.json
```

## Choosing a strategy

| Strategy | When to use | Tradeoff |
|---|---|---|
| `ListItemLineExplodeProcessor` (default) | You want **every visual line** to be its own bbox | Wrapped 2-line items get over-split |
| `ListItemGapClusterProcessor` | Source PDF uses extra spacing between items | Collapses to one item when sub-items are visually flush |

To swap, edit `marker/converters/pdf.py` and replace the entry in
`default_processors`. Both processors are prefix-free; they look at line
bboxes only.
