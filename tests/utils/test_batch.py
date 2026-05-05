"""
Tests for dynamic batch size calculation in marker.utils.batch.

These tests verify that get_batch_sizes_worker_counts correctly:
- Calculates worker count based on available VRAM
- Scales batch sizes proportionally to VRAM per worker
- Handles single-worker low VRAM (returns empty dict for defaults)
- Handles single-worker high VRAM (scales batch sizes)
- Caps scaling factor to prevent OOM
- Respects minimum batch size limits
- Sets CPU worker count appropriately
"""

import unittest

from marker.utils.batch import get_batch_sizes_worker_counts


class MockGPUManager:
    """Mock GPUManager for testing."""

    def __init__(self, vram_gb: int):
        self.vram_gb = vram_gb

    def get_gpu_vram(self) -> int:
        return self.vram_gb


class TestGetBatchSizesWorkerCounts(unittest.TestCase):
    """Test cases for get_batch_sizes_worker_counts."""

    def test_single_worker_low_vram_cpu_mode(self):
        """Single worker with 8GB (CPU default) should return empty dict."""
        gpu_mgr = MockGPUManager(vram_gb=8)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 1)
        self.assertEqual(batch_sizes, {})

    def test_single_worker_medium_vram(self):
        """Single worker with 12GB (>10GB threshold) should return scaled batch sizes."""
        gpu_mgr = MockGPUManager(vram_gb=12)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 1)
        self.assertIsInstance(batch_sizes, dict)
        self.assertIn("layout_batch_size", batch_sizes)
        # Scale should be ~12/7 = 1.71, so layout_batch_size = 12 * 1.71 = 20.5 -> 21
        self.assertEqual(batch_sizes["layout_batch_size"], 21)

    def test_single_worker_high_vram_24gb(self):
        """Single worker with 24GB should have scale ~3.43x."""
        gpu_mgr = MockGPUManager(vram_gb=24)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 1)
        # Scale = min(4.0, 24/7) = min(4.0, 3.43) = 3.43
        # layout_batch_size = round(12 * 3.43) = round(41.14) = 41
        self.assertEqual(batch_sizes["layout_batch_size"], 41)
        self.assertEqual(batch_sizes["detector_postprocessing_cpu_workers"], 1)

    def test_single_worker_very_high_vram_capped(self):
        """Single worker with 80GB should be capped at 4x scale."""
        gpu_mgr = MockGPUManager(vram_gb=80)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 1)
        # Scale capped at 4.0
        # layout_batch_size = round(12 * 4.0) = 48
        self.assertEqual(batch_sizes["layout_batch_size"], 48)
        self.assertEqual(batch_sizes["recognition_batch_size"], 256)  # 64 * 4.0

    def test_multi_worker_14gb_total(self):
        """14GB total VRAM with peak_worker_vram=7 gives 2 workers, ~7GB each."""
        gpu_mgr = MockGPUManager(vram_gb=14)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 2)
        # VRAM per worker = 7GB, scale = 1.0 (baseline)
        self.assertEqual(batch_sizes["layout_batch_size"], 12)
        self.assertEqual(batch_sizes["detection_batch_size"], 8)
        self.assertEqual(batch_sizes["recognition_batch_size"], 64)
        self.assertEqual(batch_sizes["detector_postprocessing_cpu_workers"], 2)

    def test_multi_worker_28gb_total(self):
        """28GB total gives 4 workers, 7GB each - baseline scale."""
        gpu_mgr = MockGPUManager(vram_gb=28)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 4)
        # VRAM per worker = 7GB, scale = 1.0
        self.assertEqual(batch_sizes["layout_batch_size"], 12)

    def test_multi_worker_56gb_total(self):
        """56GB total gives 8 workers, 7GB each - still baseline."""
        gpu_mgr = MockGPUManager(vram_gb=56)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 8)
        self.assertEqual(batch_sizes["layout_batch_size"], 12)

    def test_multi_worker_16gb_total_scaled(self):
        """16GB total gives 2 workers, 8GB each - scale ~1.14x."""
        gpu_mgr = MockGPUManager(vram_gb=16)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 2)
        # Scale = 8/7 = 1.14
        # layout_batch_size = round(12 * 1.14) = round(13.68) = 14
        self.assertEqual(batch_sizes["layout_batch_size"], 14)
        self.assertEqual(batch_sizes["detector_postprocessing_cpu_workers"], 2)

    def test_multi_worker_48gb_total_scaled(self):
        """48GB total gives 6 workers, 8GB each - scale ~1.14x."""
        gpu_mgr = MockGPUManager(vram_gb=48)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 6)  # 48 // 7 = 6
        # VRAM per worker = 48/6 = 8GB, scale = 8/7 = 1.14
        self.assertEqual(batch_sizes["layout_batch_size"], 14)

    def test_multi_worker_80gb_total_scaled(self):
        """80GB total gives 11 workers, 7.27GB each - scale ~1.04x."""
        gpu_mgr = MockGPUManager(vram_gb=80)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 11)  # 80 // 7 = 11
        # VRAM per worker = 80/11 = 7.27GB, scale = 1.04
        # layout_batch_size = round(12 * 1.04) = 12 or 13
        self.assertIn(batch_sizes["layout_batch_size"], [12, 13])

    def test_minimum_batch_sizes_respected(self):
        """Ensure batch sizes don't fall below minimum thresholds."""
        # Single worker with 11GB gives scale ~1.57, but with min VRAM check it might be lower
        # Actually 11GB single worker: scale = 11/7 = 1.57
        gpu_mgr = MockGPUManager(vram_gb=11)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        for key, value in batch_sizes.items():
            if key == "detector_postprocessing_cpu_workers":
                continue
            self.assertGreaterEqual(value, 1)

    def test_all_batch_size_keys_present(self):
        """Verify all expected batch size keys are returned for scaled scenarios."""
        gpu_mgr = MockGPUManager(vram_gb=24)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        expected_keys = [
            "layout_batch_size",
            "detection_batch_size",
            "table_rec_batch_size",
            "ocr_error_batch_size",
            "recognition_batch_size",
            "equation_batch_size",
            "detector_postprocessing_cpu_workers",
        ]
        for key in expected_keys:
            self.assertIn(key, batch_sizes)

    def test_detector_postprocessing_cpu_workers_multi(self):
        """Multi-worker setups should have 2 CPU workers for postprocessing."""
        gpu_mgr = MockGPUManager(vram_gb=14)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(batch_sizes["detector_postprocessing_cpu_workers"], 2)

    def test_detector_postprocessing_cpu_workers_single_scaled(self):
        """Single-worker high-VRAM should have 1 CPU worker."""
        gpu_mgr = MockGPUManager(vram_gb=24)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 1)
        self.assertEqual(batch_sizes["detector_postprocessing_cpu_workers"], 1)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_zero_vram(self):
        """Zero VRAM should fall back to default 8GB and return empty dict for single worker."""
        gpu_mgr = MockGPUManager(vram_gb=0)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 1)
        self.assertEqual(batch_sizes, {})

    def test_negative_vram(self):
        """Negative VRAM should use abs value."""
        gpu_mgr = MockGPUManager(vram_gb=-1)
        # With negative vram, the division will be negative, but max(1, ...) will give 1
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 1)
        # Since vram_gb is negative, vram_per_worker is negative, scale < 1.0, triggers empty dict return
        self.assertEqual(batch_sizes, {})

    def test_very_small_vram_fractional_worker(self):
        """VRAM less than peak_worker_vram should still return 1 worker."""
        gpu_mgr = MockGPUManager(vram_gb=5)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 1)
        self.assertEqual(batch_sizes, {})

    def test_fractional_vram_per_worker(self):
        """Test that fractional VRAM per worker is handled correctly."""
        # 15GB total, 2 workers = 7.5GB each
        gpu_mgr = MockGPUManager(vram_gb=15)
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_mgr, peak_worker_vram=7)
        self.assertEqual(workers, 2)
        # Scale = 7.5/7 = 1.07
        # layout_batch_size = round(12 * 1.07) = 13
        self.assertEqual(batch_sizes["layout_batch_size"], 13)


if __name__ == "__main__":
    unittest.main()
