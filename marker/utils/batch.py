import logging

from marker.utils.gpu import GPUManager



def get_batch_sizes_worker_counts(gpu_manager: GPUManager, peak_worker_vram: int):
    """
    Dynamically calculate optimal batch sizes and worker counts based on available VRAM.

    The function scales batch sizes proportionally to VRAM per worker, using
    reference batch sizes calibrated for ~7GB VRAM per worker (multi-worker scenario).

    For single-worker high-VRAM setups (e.g., 24GB, 48GB, 80GB), batch sizes are
    also scaled to utilize the available memory.

    Args:
        gpu_manager: GPUManager instance for VRAM detection
        peak_worker_vram: Target VRAM (in GB) per worker for baseline configuration.
                         Typically 7 GB, which accommodates peak usage of ~5GB with headroom.

    Returns:
        tuple: (dict of batch size configs, worker count)
    """
    vram_gb = gpu_manager.get_gpu_vram()

    # Calculate worker count (at least 1)
    workers = max(1, vram_gb // peak_worker_vram)

    # Reference batch sizes for ~7GB VRAM per worker (multi-worker baseline)
    # These are the current "hardcoded" values when workers > 1
    reference_batch_sizes = {
        "layout_batch_size": 12,
        "detection_batch_size": 8,
        "table_rec_batch_size": 12,
        "ocr_error_batch_size": 12,
        "recognition_batch_size": 64,
        "equation_batch_size": 16,
    }

    # Reference VRAM per worker (GB) - matches the calibration point
    reference_vram_per_worker = 7.0

    # Calculate actual VRAM per worker
    vram_per_worker = vram_gb / workers

    # Compute scaling factor (cap to reasonable range to avoid extremes)
    # Minimum scale of 1.0 ensures we never go below reference values
    # Maximum scale of 4.0 prevents overly aggressive batching that could cause OOM
    scale = vram_per_worker / reference_vram_per_worker
    scale = max(1.0, min(scale, 4.0))

    # For single worker, if we have significantly more VRAM, also scale up
    if workers == 1:
        # Check if single-worker VRAM justifies scaling (more than 10GB)
        if vram_gb > 10:
            scale = vram_gb / reference_vram_per_worker
            scale = max(1.0, min(scale, 4.0))
        else:
            # Single worker with modest VRAM - use defaults (empty dict signals callers to use their defaults)
            return {}, workers

    # Apply scaling to all batch sizes with model-specific minimums
    batch_sizes = {}
    min_batch_sizes = {
        "layout_batch_size": 2,
        "detection_batch_size": 2,
        "table_rec_batch_size": 2,
        "ocr_error_batch_size": 2,
        "recognition_batch_size": 4,
        "equation_batch_size": 4,
    }

    for key, ref_value in reference_batch_sizes.items():
        scaled_value = int(round(ref_value * scale))
        min_value = min_batch_sizes.get(key, 1)
        batch_sizes[key] = max(min_value, scaled_value)

    # Determine CPU workers for detector postprocessing
    # Use 2 for multi-worker setups, 1 for single-worker
    batch_sizes["detector_postprocessing_cpu_workers"] = 2 if workers > 1 else 1

    logger.info(
        f"Dynamic batch configuration: VRAM={vram_gb}GB, workers={workers}, "
        f"VRAM/worker={vram_per_worker:.1f}GB, scale={scale:.2f}x"
    )
    logger.debug(f"Batch sizes: {batch_sizes}")

    return batch_sizes, workers
