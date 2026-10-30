import numpy as np

from marker.processors.sectionheader import SectionHeaderProcessor


class FakeKmeans:
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, **kwargs):
        return self

    def fit_predict(self, data):
        return np.asarray(self.labels)


def test_bucket_headings_sorts_rows_by_height_keeping_label_pairs(monkeypatch):
    # Unequal cluster sizes and labels ordered opposite to their heights
    # expose column-wise sorting: it would group [12, 14] and [16, 22, 26]
    # instead of preserving the original height/label pairs.
    monkeypatch.setattr(
        "marker.processors.sectionheader.KMeans",
        FakeKmeans([1, 0, 1, 1, 0]),
    )
    processor = SectionHeaderProcessor({"level_count": 2})

    heading_ranges = processor.bucket_headings([12, 22, 14, 16, 26])

    assert heading_ranges == [(22, 26), (12, 16)]


def test_bucket_headings_clamps_cluster_count_to_distinct_sizes():
    # Six heights, but only two distinct sizes once rounding accounts for
    # rendering jitter. Without clamping, n_clusters=4 invents heading levels.
    processor = SectionHeaderProcessor({"level_count": 4})

    heading_ranges = processor.bucket_headings([11.9, 12.0, 12.1, 17.9, 18.0, 18.1])

    assert len(heading_ranges) == 2


def test_bucket_headings_returns_empty_when_only_one_distinct_size():
    processor = SectionHeaderProcessor({"level_count": 2})

    heading_ranges = processor.bucket_headings([10.0, 10.1, 10.2, 10.3, 10.4])

    assert heading_ranges == []


def test_bucket_headings_returns_empty_when_few_lines_than_levels():
    processor = SectionHeaderProcessor({"level_count": 4})

    assert processor.bucket_headings([10, 12, 14]) == []
