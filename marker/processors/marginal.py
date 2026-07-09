"""
Processor for detecting and assigning marginal numbers. 

Marginal numbers are used e.g. in legal texts. Each number is associated with 
the paragraph whose first line sits at the same vertical position as the 
number. The processor inserts a Marginal block as the first child of the 
associated paragraph so the renderer can emit <aside>#</aside> before the 
paragraph text.

Detection runs in two passes: block-level and span-level.

Association algorithm:
  mod_baseline = source.polygon.y_start  (SHIFT_BASELINE_FACTOR is 0.0)

  1. CONTAINMENT: pick the non-mega candidate whose top_y is closest to
     mod_baseline from above (largest top_y among containing blocks).
  2. LOWER: nearest candidate below mod_baseline, prefer larger bottom_y.
  3. UPPER: nearest candidate above mod_baseline, prefer smaller top_y.
  Decision: lower wins unless upper_delta <= 2*lh and lower_delta > 1.5*upper_delta.
"""

import re
from collections import defaultdict
from statistics import median
from typing import Annotated, Dict, List, Optional, Set, Tuple

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.blocks.marginal import Marginal
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.logger import get_logger

logger = get_logger()

_ASSOCIATED_BLOCK_TYPES = (
    BlockTypes.Text,
    BlockTypes.SectionHeader,
    BlockTypes.ListItem,
    BlockTypes.ListGroup,
)

_BLOCK_CANDIDATE_TYPES = (
    BlockTypes.Text,
    BlockTypes.TextInlineMath,
    BlockTypes.SectionHeader,
    BlockTypes.ListItem,
)

_SPAN_SCAN_BLOCK_TYPES = (
    BlockTypes.Text,
    BlockTypes.TextInlineMath,
    BlockTypes.SectionHeader,
    BlockTypes.ListItem,
    BlockTypes.ListGroup,
)

_LIST_CONTAINER_TYPES = (BlockTypes.ListItem, BlockTypes.ListGroup)

SHIFT_BASELINE_FACTOR = 0.0


def _rn_sort_key(rn_text: str) -> tuple:
    m = re.match(r'^\s*(\d{1,4})([a-zA-Z]{0,2})(?:[.\-](\d{1,2}))?\s*$', rn_text)
    if not m:
        return (0, 0, 0)
    num = int(m.group(1))
    alpha = m.group(2) or ''
    sub = m.group(3)
    if sub:
        return (num, 2, int(sub))
    elif alpha:
        return (num, 1, ord(alpha[-1].lower()))
    else:
        return (num, 0, 0)


class MarginalProcessor(BaseProcessor):
    """Detects Randnummern and inserts them as <aside>N</aside> before their paragraph."""

    block_types = (BlockTypes.Text,)

    min_text_block_width_ratio: Annotated[
        float,
        "Blocks wider than this fraction of page width define the text column.",
    ] = 0.50

    scan_epsilon_ratio: Annotated[
        float,
        "Margin zone tolerance as fraction of page width.",
    ] = 0.02

    min_marginal_pages_threshold: Annotated[
        int,
        "Min pages with marginals on both sides to activate two-column mode.",
    ] = 5

    marginal_pages_ratio_threshold: Annotated[
        float,
        "Fraction of marginal-pages needing both sides for two-column mode.",
    ] = 0.25

    max_contain_height_ratio: Annotated[
        float,
        "Blocks taller than this fraction of page height are treated as mega-blocks "
        "and deprioritised in the containment step. Set to 1.0 to disable.",
    ] = 0.95

    MARGINAL_PATTERN = re.compile(r'^\s*\d{1,4}[a-zA-Z]{0,2}([.\-]\d{1,2})?\s*$')

    def _matches(self, text: str) -> bool:
        return bool(self.MARGINAL_PATTERN.match(text.strip()))

    def _page_line_height(self, page: PageGroup, document: Document) -> float:
        cached = self._lh_cache.get(page.page_id)
        if cached is not None:
            return cached
        heights = []
        for block in (page.children or []):
            try:
                for line in block.contained_blocks(document, (BlockTypes.Line,)):
                    h = line.polygon.height
                    if 4 < h < 40:
                        heights.append(h)
            except Exception:
                pass
        lh = median(heights) if heights else 9.0
        self._lh_cache[page.page_id] = lh
        return lh

    def _compute_text_column(
        self, document: Document
    ) -> Tuple[Optional[float], Optional[float]]:
        x_starts: List[float] = []
        x_ends: List[float] = []
        for page in document.pages:
            min_w = page.polygon.width * self.min_text_block_width_ratio
            for block in page.contained_blocks(
                document, (BlockTypes.Text, BlockTypes.TextInlineMath)
            ):
                if not block.ignore_for_output and block.polygon.width >= min_w:
                    x_starts.append(block.polygon.x_start)
                    x_ends.append(block.polygon.x_end)
        if len(x_starts) < 3:
            return None, None
        return median(x_starts), median(x_ends)

    def _in_left_margin(self, x_end: float, tx_start: float, eps: float) -> bool:
        return x_end < tx_start + eps

    def _in_right_margin(self, x_start: float, tx_end: float, eps: float) -> bool:
        return x_start > tx_end - eps

    @staticmethod
    def _get_baseline_y(block: Block, document: Document) -> float:
        try:
            origin = getattr(block, "origin", None)
            if origin is not None and len(origin) >= 2:
                return float(origin[1])
            spans = block.contained_blocks(document, (BlockTypes.Span,))
            if spans:
                origin = getattr(spans[0], "origin", None)
                if origin is not None and len(origin) >= 2:
                    return float(origin[1])
            lines = block.contained_blocks(document, (BlockTypes.Line,))
            if lines:
                lb = lines[0].polygon
                return lb.y_end - 0.20 * lb.height
        except Exception:
            pass
        bp = block.polygon
        return bp.y_end - 0.20 * bp.height

    def _detect_block_marginals(
        self, page: PageGroup, document: Document,
        tx_start: float, tx_end: float, eps: float,
    ) -> Tuple[List, List]:
        left, right = [], []
        if not page.children:
            return left, right
        for block in page.children:
            if block.block_type not in _BLOCK_CANDIDATE_TYPES or block.ignore_for_output:
                continue
            text = block.raw_text(document).strip()
            if not self._matches(text):
                continue
            x0, x1 = block.polygon.x_start, block.polygon.x_end
            if self._in_left_margin(x1, tx_start, eps):
                left.append((block, text))
                logger.debug(f"Block marginal L '{text}' x={x0:.0f}-{x1:.0f} p{page.page_id}")
            elif self._in_right_margin(x0, tx_end, eps):
                right.append((block, text))
                logger.debug(f"Block marginal R '{text}' x={x0:.0f}-{x1:.0f} p{page.page_id}")
        return left, right

    def _detect_span_marginals(
        self, page: PageGroup, document: Document,
        tx_start: float, tx_end: float, eps: float,
    ) -> Tuple[List, List]:
        left, right = [], []
        if not page.children:
            return left, right
        page_width = page.polygon.width
        for block in page.children:
            if block.block_type not in _SPAN_SCAN_BLOCK_TYPES or block.ignore_for_output:
                continue
            # Skip wide Text/TextInlineMath blocks starting inside the text column:
            # a leading "1" in a full-column paragraph is a footnote number, not a Rn.
            # ListItem/ListGroup/SectionHeader are exempt.
            skip_left = (
                block.block_type in (BlockTypes.Text, BlockTypes.TextInlineMath)
                and block.polygon.width > page_width * 0.50
                and block.polygon.x_start >= tx_start - eps
            )
            block_lines = list(block.contained_blocks(document, (BlockTypes.Line,)))
            num_lines = len(block_lines)
            for line_idx, line in enumerate(block_lines):
                is_last_line = (line_idx == num_lines - 1)
                spans = line.contained_blocks(document, (BlockTypes.Span,))
                if not spans:
                    continue
                if not skip_left:
                    first = spans[0]
                    ft = first.text.strip()
                    if self._matches(ft) and \
                            self._in_left_margin(first.polygon.x_end, tx_start, eps):
                        left.append((first, ft, line, block))
                        logger.debug(f"Span marginal L '{ft}' p{page.page_id}")
                        continue
                last = spans[-1]
                lt = last.text.strip()
                if self._matches(lt) and \
                        self._in_right_margin(last.polygon.x_start, tx_end, eps):
                    right.append((last, lt, line, block))
                    logger.debug(f"Span marginal R '{lt}' p{page.page_id}")
                    continue
                # OCR sometimes merges a trailing right-margin Rn into the last span
                # of the final line. Only check the last line.
                if is_last_line and lt and block.polygon.x_end > tx_end:
                    trailing = lt.split()[-1]
                    if self._matches(trailing) and not self._matches(lt):
                        right.append((last, trailing, line, block))
                        logger.debug(
                            f"Span marginal R (embedded) '{trailing}' p{page.page_id}"
                        )
        return left, right

    @staticmethod
    def _deduplicate_block_vs_span(
        block_list: List[Tuple], span_list: List[Tuple], document: Document,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Resolve block/span conflicts for the same marginal on the same side.

        If the span's parent is the same standalone block as the block entry,
        keep the block entry and drop the span (case A). Otherwise drop the
        block entry and keep the span, which carries richer context (case B).
        """
        if not block_list or not span_list:
            return block_list, span_list

        span_entries_by_text: Dict[str, List] = {}
        for entry in span_list:
            span_entries_by_text.setdefault(entry[1], []).append(entry)

        kept_blocks: List[Tuple] = []
        suppressed_span_texts: set = set()

        for block, b_text in block_list:
            entries = span_entries_by_text.get(b_text, [])
            if not entries:
                kept_blocks.append((block, b_text))
                continue
            span_parent_ids = {e[3].id for e in entries}
            if span_parent_ids == {block.id}:
                # Case A: span comes from the same standalone block
                kept_blocks.append((block, b_text))
                suppressed_span_texts.add(b_text)
                logger.debug(f"Dedup: kept BL '{b_text}', suppressed SL from same block")
            else:
                # Case B: span from a different block wins
                logger.debug(f"Dedup: suppressed BL '{b_text}' (SL on different block)")

        if suppressed_span_texts:
            filtered_spans = [e for e in span_list if e[1] not in suppressed_span_texts]
        else:
            filtered_spans = span_list

        return kept_blocks, filtered_spans

    @staticmethod
    def _deduplicate_span_vs_span(
        span_list: List[Tuple], document: Document,
    ) -> List[Tuple]:
        """When two spans share the same marginal, keep the one from a non-pure-marginal block."""
        if not span_list:
            return span_list
        by_text: Dict[str, List] = defaultdict(list)
        for entry in span_list:
            by_text[entry[1]].append(entry)

        result = []
        for text, entries in by_text.items():
            if len(entries) == 1:
                result.append(entries[0])
                continue
            non_pure = [e for e in entries if e[3].raw_text(document).strip() != text]
            if non_pure:
                non_pure.sort(key=lambda e: e[0].polygon.y_start)
                kept = non_pure[0]
            else:
                entries.sort(key=lambda e: e[0].polygon.y_start)
                kept = entries[0]
            result.append(kept)
            dropped = len(entries) - 1
            if dropped:
                logger.debug(
                    f"Span dedup: kept 1 of {len(entries)} spans for '{text}', "
                    f"dropped {dropped} pure-Rn span(s)"
                )
        return result

    def detect_marginals_per_page(
        self, document: Document, tx_start: float, tx_end: float,
    ) -> Dict[int, Dict]:
        result = {}
        for page in document.pages:
            eps = page.polygon.width * self.scan_epsilon_ratio
            bl, br = self._detect_block_marginals(page, document, tx_start, tx_end, eps)
            sl, sr = self._detect_span_marginals(page, document, tx_start, tx_end, eps)
            bl, sl = self._deduplicate_block_vs_span(bl, sl, document)
            br, sr = self._deduplicate_block_vs_span(br, sr, document)
            sl = self._deduplicate_span_vs_span(sl, document)
            sr = self._deduplicate_span_vs_span(sr, document)
            # Cross-side dedup: prefer left-side entry when the same marginal appears on both.
            sl_texts = {e[1] for e in sl}
            bl_texts = {e[1] for e in bl}
            removed_sr = [e for e in sr if e[1] in sl_texts or e[1] in bl_texts]
            sr = [e for e in sr if e[1] not in sl_texts and e[1] not in bl_texts]
            for e in removed_sr:
                span = e[0]
                span_txt = span.text.strip()
                rn_txt = e[1]
                if span_txt == rn_txt:
                    span.ignore_for_output = True
                else:
                    stripped = span_txt[: span_txt.rfind(rn_txt)].rstrip()
                    leading_ws = span.text[: len(span.text) - len(span.text.lstrip())]
                    span.text = leading_ws + stripped if stripped else span.text
            br_texts = {e[1] for e in br}
            sl = [e for e in sl if e[1] not in br_texts]
            if bl or br or sl or sr:
                result[page.page_id] = {
                    "block_left": bl, "block_right": br,
                    "span_left": sl, "span_right": sr,
                }
        return result

    def detect_two_column_mode(self, page_marginals: Dict) -> bool:
        total = len(page_marginals)
        if not total:
            return False
        both = sum(
            1 for v in page_marginals.values()
            if (v["block_left"] or v["span_left"])
            and (v["block_right"] or v["span_right"])
        )
        return (
            both >= self.min_marginal_pages_threshold
            or both / total > self.marginal_pages_ratio_threshold
        )

    def _find_associated_block(
        self, source: Block, page: PageGroup, document: Document,
        two_column: bool, side: str, claimed: Set,
        lh: float = 0.0,
        candidates: Optional[List[Block]] = None,
    ) -> Optional[Block]:
        if lh == 0.0:
            lh = self._page_line_height(page, document)

        mod_baseline = source.polygon.y_start + SHIFT_BASELINE_FACTOR * lh
        page_width = page.polygon.width

        raw_candidates = candidates if candidates is not None else \
            list(page.contained_blocks(document, _ASSOCIATED_BLOCK_TYPES))

        filtered: List[Block] = []
        for block in raw_candidates:
            if block.ignore_for_output or block.id in claimed or block.id == source.id:
                continue
            btype = block.block_type
            min_w_ratio = 0.05 if btype in (
                BlockTypes.SectionHeader, BlockTypes.ListItem, BlockTypes.ListGroup
            ) else 0.10
            if block.polygon.width < page_width * min_w_ratio:
                continue
            if two_column:
                mid = page_width / 2
                cx = block.polygon.x_start + block.polygon.width / 2
                if side == "left" and cx >= mid:
                    continue
                if side == "right" and cx < mid:
                    continue
            filtered.append(block)

        # Containment: baseline falls inside the block (0.5 px upward tolerance)
        containing = [
            b for b in filtered
            if (b.polygon.y_start - 0.5) <= mod_baseline <= b.polygon.y_end
        ]
        if containing:
            max_h = page.polygon.height * self.max_contain_height_ratio
            preferred = [b for b in containing if b.polygon.height <= max_h]
            pool = preferred if preferred else containing
            return max(pool, key=lambda b: b.polygon.y_start)

        # Lower candidates
        lower_cands = [b for b in filtered if b.polygon.y_start > mod_baseline]
        lower_block: Optional[Block] = None
        lower_delta: float = 0.0
        if lower_cands:
            lower_delta = min(b.polygon.y_start - mod_baseline for b in lower_cands)
            closest = [b for b in lower_cands if (b.polygon.y_start - mod_baseline) == lower_delta]
            max_bottom = max(b.polygon.y_end for b in closest)
            lower_block = [b for b in closest if b.polygon.y_end == max_bottom][0]

        # Upper candidates
        upper_cands = [b for b in filtered if b.polygon.y_end < mod_baseline]
        upper_block: Optional[Block] = None
        upper_delta: float = 0.0
        if upper_cands:
            upper_delta = min(mod_baseline - b.polygon.y_end for b in upper_cands)
            closest = [b for b in upper_cands if (mod_baseline - b.polygon.y_end) == upper_delta]
            min_top = min(b.polygon.y_start for b in closest)
            upper_block = [b for b in closest if b.polygon.y_start == min_top][-1]

        if lower_block is None and upper_block is None:
            logger.debug(
                f"_find_associated_block: no candidate at "
                f"mod_baseline={mod_baseline:.1f} p{page.page_id}"
            )
            return None
        if lower_block is None:
            return upper_block
        if upper_block is None:
            return lower_block
        if upper_delta > 2 * lh:
            return lower_block
        if lower_delta > upper_delta * 1.5:
            return upper_block
        return lower_block

    def _find_containing_list_item(
        self, line: Block, parent_block: Block, document: Document,
    ) -> Optional[Block]:
        if parent_block.block_type == BlockTypes.ListItem:
            return parent_block
        if parent_block.block_type == BlockTypes.ListGroup:
            for item in parent_block.contained_blocks(document, (BlockTypes.ListItem,)):
                if item.structure and line.id in item.structure:
                    return item
        return None

    def _make_marginal(
        self, source: Block, page: PageGroup, text: str, assoc: Block,
    ) -> Marginal:
        src_poly = source.polygon
        y0, y1, x1 = src_poly.y_start, src_poly.y_end, src_poly.x_end
        # x_start=0 ensures SectionHeader's x_start sort places aside before heading text
        poly = PolygonBox(polygon=[[0, y0], [x1, y0], [x1, y1], [0, y1]])
        return Marginal(
            polygon=poly,
            page_id=page.page_id,
            marginal_number=text,
            associated_block_id=str(assoc.id),
            text_extraction_method=getattr(source, "text_extraction_method", None),
            structure=None,
            source="processor",
        )

    def _fix_inverted_assignments(
        self, marginals: List["Marginal"], document: Document,
        block_cache: Optional[Dict[str, Block]] = None,
        page_heights: Optional[Dict[int, float]] = None,
        page_cache: Optional[Dict[int, PageGroup]] = None,
    ) -> None:
        """Swap marginal assignments where an ancestor block carries a larger marginal than its descendant."""
        if len(marginals) < 2:
            return

        if block_cache is None:
            block_cache = {}
        if page_heights is None:
            page_heights = {}
        if page_cache is None:
            page_cache = {}

        for m in marginals:
            bid = m.associated_block_id
            if bid and bid not in block_cache:
                for page in document.pages:
                    if page.page_id == m.page_id:
                        page_heights.setdefault(m.page_id, page.polygon.height)
                        page_cache.setdefault(m.page_id, page)
                        for blk in page.contained_blocks(document, _ASSOCIATED_BLOCK_TYPES):
                            block_cache[str(blk.id)] = blk
                        break

        max_h_ratio = self.max_contain_height_ratio

        def is_mega(blk: Block, page_id: int) -> bool:
            return blk.polygon.height > page_heights.get(page_id, 9999.0) * max_h_ratio

        def geom_ancestor(b_anc: Block, b_desc: Block, page_id: int) -> bool:
            if is_mega(b_anc, page_id) or is_mega(b_desc, page_id):
                return False
            return (
                b_anc.polygon.y_start <= b_desc.polygon.y_start
                and b_anc.polygon.y_end >= b_desc.polygon.y_end
                and b_anc.id != b_desc.id
            )

        for _ in range(len(marginals)):
            made_swap = False
            m_list = list(marginals)
            for i, m_a in enumerate(m_list):
                if m_a.associated_block_id is None:
                    continue
                b_a = block_cache.get(m_a.associated_block_id)
                if b_a is None:
                    continue
                for m_b in m_list[i + 1:]:
                    if m_b.associated_block_id is None:
                        continue
                    b_b = block_cache.get(m_b.associated_block_id)
                    if b_b is None:
                        continue
                    pid = m_a.page_id
                    if geom_ancestor(b_a, b_b, pid):
                        anc_m, desc_m, anc_b, desc_b = m_a, m_b, b_a, b_b
                    elif geom_ancestor(b_b, b_a, pid):
                        anc_m, desc_m, anc_b, desc_b = m_b, m_a, b_b, b_a
                    else:
                        continue
                    if _rn_sort_key(anc_m.marginal_number) <= _rn_sort_key(desc_m.marginal_number):
                        continue
                    logger.debug(
                        f"Rn inversion: {anc_m.marginal_number}->ancestor, "
                        f"{desc_m.marginal_number}->descendant — swapping"
                    )
                    if anc_b.structure and anc_m.id in anc_b.structure:
                        anc_b.structure.remove(anc_m.id)
                    if desc_b.structure and desc_m.id in desc_b.structure:
                        desc_b.structure.remove(desc_m.id)
                    anc_m.associated_block_id = str(desc_b.id)
                    desc_m.associated_block_id = str(anc_b.id)
                    _page = page_cache.get(anc_m.page_id)
                    self._attach(anc_m, desc_b, _page, document)
                    self._attach(desc_m, anc_b, _page, document)
                    made_swap = True
            if not made_swap:
                break

    def _attach(self, marginal: "Marginal", assoc: Block,
                page: Optional[PageGroup] = None,
                document: Optional[Document] = None) -> None:
        """Insert marginal.id at the correct position in assoc.structure."""
        if assoc.structure is None:
            assoc.structure = []

        if page is not None and document is not None:
            page_h = page.polygon.height
            if assoc.polygon.height > page_h * self.max_contain_height_ratio:
                # Mega-block: insert before the child that spatially contains the Rn baseline.
                structure_str = {str(bid) for bid in (assoc.structure or [])}
                block_children = sorted(
                    [b for b in assoc.contained_blocks(document, _ASSOCIATED_BLOCK_TYPES)
                     if str(b.id) in structure_str],
                    key=lambda b: b.polygon.y_start,
                )
                if block_children:
                    rn_y = marginal.polygon.y_start
                    containing = [
                        b for b in block_children
                        if b.polygon.y_start <= rn_y <= b.polygon.y_end
                    ]
                    if containing:
                        target = min(containing, key=lambda b: b.polygon.y_start)
                    else:
                        max_child_end = max(b.polygon.y_end for b in block_children)
                        if rn_y > max_child_end:
                            target = None
                        else:
                            after = [b for b in block_children if b.polygon.y_start > rn_y]
                            target = after[0] if after else None

                    if target is not None:
                        nc_id_str = str(target.id)
                        idx = next(
                            (i for i, bid in enumerate(assoc.structure)
                             if str(bid) == nc_id_str),
                            None,
                        )
                        if idx is not None:
                            assoc.structure.insert(idx, marginal.id)
                        else:
                            assoc.structure.append(marginal.id)
                        logger.debug(
                            f"Mega-attach Rn{marginal.marginal_number}: "
                            f"before child y={int(target.polygon.y_start)}"
                        )
                    else:
                        assoc.structure.append(marginal.id)
                        logger.debug(f"Mega-attach Rn{marginal.marginal_number}: appended")
                    return

        assoc.structure.insert(0, marginal.id)

    def _process_block_marginals(
        self, side_list: List[Tuple], side: str,
        page: PageGroup, document: Document, two_column: bool,
        to_remove: List, claimed: Set,
        lh: float = 0.0,
        candidates: Optional[List[Block]] = None,
    ) -> None:
        for source_block, text in side_list:
            assoc = self._find_associated_block(
                source_block, page, document, two_column, side, claimed,
                lh=lh, candidates=candidates,
            )
            if not assoc:
                source_block.ignore_for_output = True
                source_block.structure = None
                if source_block.id in page.structure:
                    to_remove.append(source_block.id)
                logger.debug(f"No target for block marginal '{text}' p{page.page_id}")
                continue
            claimed.add(assoc.id)
            m = self._make_marginal(source_block, page, text, assoc)
            page.replace_block(source_block, m)
            if m.id in page.structure:
                to_remove.append(m.id)
            elif source_block.id in page.structure:
                to_remove.append(source_block.id)
            self._attach(m, assoc, page, document)
            logger.debug(f"Block marginal '{text}' -> {assoc.id} p{page.page_id}")

    def _process_span_marginals(
        self, side_list: List[Tuple], side: str,
        page: PageGroup, document: Document, two_column: bool,
        claimed: Set,
        lh: float = 0.0,
        candidates: Optional[List[Block]] = None,
    ) -> None:
        for span, text, parent_line, parent_block in side_list:
            assoc = self._find_associated_block(
                span, page, document, two_column, side, claimed,
                lh=lh, candidates=candidates,
            )
            if not assoc and parent_block.block_type in _LIST_CONTAINER_TYPES:
                li = self._find_containing_list_item(parent_line, parent_block, document)
                if li and li.id not in claimed:
                    assoc = li
                    logger.debug(f"Span '{text}' -> ListItem fallback {li.id} p{page.page_id}")
            if not assoc:
                span.ignore_for_output = True
                logger.debug(f"No target for span marginal '{text}' p{page.page_id}")
                continue
            claimed.add(assoc.id)
            span_text_full = span.text.strip()
            embedded = (span_text_full != text)
            if not embedded:
                if parent_line.structure:
                    parent_line.structure = [
                        bid for bid in parent_line.structure if bid != span.id
                    ]
                span.ignore_for_output = True
            else:
                stripped = span_text_full[: span_text_full.rfind(text)].rstrip()
                leading_ws = span.text[: len(span.text) - len(span.text.lstrip())]
                span.text = leading_ws + stripped if stripped else span.text
            m = self._make_marginal(span, page, text, assoc)
            page.add_full_block(m)
            self._attach(m, assoc, page, document)
            logger.debug(f"Span marginal '{text}' -> {assoc.id} p{page.page_id}")
            if m.id in page.structure:
                page.remove_structure_items([m.id])

    def __call__(self, document: Document) -> None:
        self._lh_cache: Dict[int, float] = {}

        tx_start, tx_end = self._compute_text_column(document)
        if tx_start is None:
            logger.debug("MarginalProcessor: text column undetermined, skipping")
            return
        logger.debug(f"MarginalProcessor: text column x=[{tx_start:.1f}, {tx_end:.1f}]")

        page_marginals = self.detect_marginals_per_page(document, tx_start, tx_end)
        if not page_marginals:
            logger.debug("MarginalProcessor: no marginals found")
            return

        two_col = self.detect_two_column_mode(page_marginals)
        logger.info(
            f"MarginalProcessor: {'two-column' if two_col else 'single-column'}, "
            f"{len(page_marginals)} pages with marginals"
        )

        shared_block_cache: Dict[str, Block] = {}
        shared_page_heights: Dict[int, float] = {}
        shared_page_cache: Dict[int, PageGroup] = {}

        for page in document.pages:
            info = page_marginals.get(page.page_id)
            if not info:
                continue

            lh = self._page_line_height(page, document)
            page_candidates = list(page.contained_blocks(document, _ASSOCIATED_BLOCK_TYPES))

            shared_page_heights[page.page_id] = page.polygon.height
            shared_page_cache[page.page_id] = page
            for blk in page_candidates:
                shared_block_cache[str(blk.id)] = blk

            to_remove: List = []
            claimed: Set = set()
            self._process_block_marginals(
                info["block_left"], "left", page, document, two_col, to_remove, claimed,
                lh=lh, candidates=page_candidates)
            self._process_block_marginals(
                info["block_right"], "right", page, document, two_col, to_remove, claimed,
                lh=lh, candidates=page_candidates)
            self._process_span_marginals(
                info["span_left"], "left", page, document, two_col, claimed,
                lh=lh, candidates=page_candidates)
            self._process_span_marginals(
                info["span_right"], "right", page, document, two_col, claimed,
                lh=lh, candidates=page_candidates)

            page_marginals_created = [
                blk for blk in page.contained_blocks(document, (BlockTypes.Marginal,))
                if getattr(blk, "source", None) == "processor"
            ]
            if page_marginals_created:
                self._fix_inverted_assignments(
                    page_marginals_created, document,
                    block_cache=shared_block_cache,
                    page_heights=shared_page_heights,
                    page_cache=shared_page_cache,
                )

            if to_remove:
                page.remove_structure_items(to_remove)
                logger.debug(f"Removed {len(to_remove)} from page.structure p{page.page_id}")
