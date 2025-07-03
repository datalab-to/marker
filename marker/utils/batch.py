from marker.utils.gpu import GPUManager


def get_batch_sizes_worker_counts(gpu_manager: GPUManager, peak_worker_vram: int):
    vram = gpu_manager.get_gpu_vram()

    workers = max(1, vram // peak_worker_vram)
    if workers == 1:
        return {}, workers

    return {
        "layout_batch_size": 12,
        "detection_batch_size": 8,
        "table_rec_batch_size": 12,
        "ocr_error_batch_size": 12,
        "recognition_batch_size": 64,
        "equation_batch_size": 16,
        "detector_postprocessing_cpu_workers": 2,
    }, workers
