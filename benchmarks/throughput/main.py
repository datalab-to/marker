import os
import tempfile
import time
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
import torch

import click
import pypdfium2 as pdfium
from tqdm import tqdm

import datasets


def get_next_pdf(ds: datasets.Dataset, i: int):
    while True:
        pdf = ds[i]["pdf"]
        filename = ds[i]["filename"]
        if pdf and filename.endswith(".pdf"):
            return pdf, filename, i + 1
        i += 1
        if i >= len(ds):
            i = 0


def single_batch(
    batch_size: int,
    num_threads: int,
    force_ocr: bool,
    quantize: bool,
    compile: bool,
    worker_id: int,
    chunksize: int = 100,
):
    if quantize:
        os.environ["RECOGNITION_MODEL_QUANTIZE"] = "true"
    if compile:
        os.environ["COMPILE_ALL"] = "true"

    for item in [
        "DETECTOR_POSTPROCESSING_CPU_WORKERS",
        "OPENBLAS_NUM_THREADS",
        "PDFTEXT_CPU_WORKERS",
        "OMP_NUM_THREADS",
    ]:
        os.environ[item] = f"{num_threads}"

    torch.set_num_threads(num_threads)

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    ds = datasets.load_dataset("datalab-to/pdfs", split="train")
    model_dict = create_model_dict()
    torch.cuda.reset_peak_memory_stats()

    times = []
    i = 0
    pages = 0
    chars = 0

    min_time = time.time()
    for _ in range(batch_size):
        pdf, fname, i = get_next_pdf(ds, i)
        print(f"Inferencing {fname} on worker {worker_id}...")

        pdf_doc = pdfium.PdfDocument(pdf)
        page_count = len(pdf_doc)
        pdf_doc.close()
        pages += page_count

        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(pdf)
            f.flush()
            page_range_chunks = list(range(0, page_count, chunksize))
            for chunk_start in page_range_chunks:
                chunk_end = min(chunk_start + chunksize, page_count)
                page_range = list(range(chunk_start, chunk_end))

                block_converter = PdfConverter(
                    artifact_dict=model_dict,
                    config={
                        "disable_tqdm": worker_id > 0,
                        "page_range": page_range,
                        "force_ocr": force_ocr,
                    },
                )
                start = time.time()
                rendered = block_converter(f.name)
                markdown, _, _ = text_from_rendered(rendered)
                chars += len(markdown)

                total = time.time() - start
                times.append(total)

    max_gpu_vram = torch.cuda.max_memory_reserved() / 1024**3
    max_time = time.time()
    return sum(times), min_time, max_time, max_gpu_vram, pages, chars


@click.command(help="Benchmark PDF to MD conversion throughput.")
@click.option("--workers", default=1, help="Number of workers to use.")
@click.option("--batch_size", default=1, help="Batch size for inference.")
@click.option("--force_ocr", is_flag=True, help="Force OCR on all pages.")
@click.option("--quantize", is_flag=True, help="Use quantized model.")
@click.option("--compile", is_flag=True, help="Use compiled model.")
def main(
    workers: int,
    batch_size: int,
    force_ocr: bool,
    quantize: bool,
    compile: bool,
):
    total_cpus = os.cpu_count()
    start = time.time()
    current_gpu_vram = torch.cuda.memory_reserved() / 1024**3
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as executor:
        cpus_per_worker = min(8, max(2, total_cpus // workers))
        futures = [
            executor.submit(
                single_batch,
                batch_size,
                cpus_per_worker,
                force_ocr,
                quantize,
                compile,
                i,
            )
            for i in range(workers)
        ]
        all_times = []
        min_time = None
        max_time = time.time()
        vrams = []
        page_count = 0
        char_count = 0
        for future in tqdm(futures, desc="Running marker workers..."):
            times, min_time_worker, max_time_worker, max_vram, pages, chars = (
                future.result()
            )
            vrams.append(max_vram - current_gpu_vram)
            all_times.append(times)
            page_count += pages
            char_count += chars
            min_time = (
                min(min_time_worker, min_time)
                if min_time is not None
                else min_time_worker
            )
            max_time = max(max_time, max_time_worker)

    end = time.time() - start
    all_worker_time = max_time - min_time

    print(f"Average time per worker: {sum(all_times) / len(all_times)}")
    print(f"Max time per worker: {max(all_times)}")
    print(f"End to end time (counting model loading), all processes: {end}")
    print(f"End to end time (no model loading), all processes: {all_worker_time}")
    print(f"Total pages: {page_count}")
    print(f"Total characters: {char_count}")
    print(f"Time per page: {all_worker_time / page_count:.2f}")
    print(f"Characters per second: {char_count / all_worker_time:.2f}")
    print(f"Max GPU VRAM: {max(vrams):.2f} GB")
    print(f"Average GPU VRAM: {sum(vrams) / len(vrams):.2f} GB")


if __name__ == "__main__":
    main()
