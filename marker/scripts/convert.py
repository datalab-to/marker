import atexit
import os
import time

import psutil
import torch

from marker.utils.batch import get_batch_sizes_worker_counts

# Ensure threads don't contend
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "2"  # Avoid OpenMP issues with multiprocessing
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Transformers uses .isin for a simple op, which is not supported on MPS
)
os.environ["IN_STREAMLIT"] = "true"  # Avoid multiprocessing inside surya

import math
import traceback

import click
import torch.multiprocessing as mp
from tqdm import tqdm
import gc

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import output_exists, save_output
from marker.utils.gpu import GPUManager

configure_logging()
logger = get_logger()


def worker_init():
    model_dict = create_model_dict()

    global model_refs
    model_refs = model_dict

    # Ensure we clean up the model references on exit
    atexit.register(worker_exit)


def worker_exit():
    global model_refs
    try:
        del model_refs
    except Exception:
        pass


def process_single_pdf(args):
    page_count = 0
    fpath, cli_options = args
    torch.set_num_threads(cli_options["total_torch_threads"])
    del cli_options["total_torch_threads"]

    config_parser = ConfigParser(cli_options)

    out_folder = config_parser.get_output_folder(fpath)
    base_name = config_parser.get_base_filename(fpath)
    if cli_options.get("skip_existing") and output_exists(out_folder, base_name):
        return page_count

    converter_cls = config_parser.get_converter_cls()
    config_dict = config_parser.generate_config_dict()
    config_dict["disable_tqdm"] = True

    try:
        if cli_options.get("debug_print"):
            logger.debug(f"Converting {fpath}")
        converter = converter_cls(
            config=config_dict,
            artifact_dict=model_refs,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(fpath)
        out_folder = config_parser.get_output_folder(fpath)
        save_output(rendered, out_folder, base_name)
        page_count = converter.page_count

        if cli_options.get("debug_print"):
            logger.debug(f"Converted {fpath}")
        del rendered
        del converter
    except Exception as e:
        logger.error(f"Error converting {fpath}: {e}")
        traceback.print_exc()
    finally:
        gc.collect()

    return page_count


@click.command(cls=CustomClickPrinter)
@click.argument("in_folder", type=str)
@click.option("--chunk_idx", type=int, default=0, help="Chunk index to convert")
@click.option(
    "--num_chunks",
    type=int,
    default=1,
    help="Number of chunks being processed in parallel",
)
@click.option(
    "--max_files", type=int, default=None, help="Maximum number of pdfs to convert"
)
@click.option(
    "--skip_existing",
    is_flag=True,
    default=False,
    help="Skip existing converted files.",
)
@click.option(
    "--debug_print", is_flag=True, default=False, help="Print debug information."
)
@click.option(
    "--max_tasks_per_worker",
    type=int,
    default=10,
    help="Maximum number of tasks per worker process before recycling.",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of worker processes to use.  Set automatically by default, but can be overridden.",
)
@ConfigParser.common_options
def convert_cli(in_folder: str, **kwargs):
    total_pages = 0
    in_folder = os.path.abspath(in_folder)
    files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
    files = [f for f in files if os.path.isfile(f)]

    # Handle chunks if we're processing in parallel
    # Ensure we get all files into a chunk
    chunk_size = math.ceil(len(files) / kwargs["num_chunks"])
    start_idx = kwargs["chunk_idx"] * chunk_size
    end_idx = start_idx + chunk_size
    files_to_convert = files[start_idx:end_idx]

    # Limit files converted if needed
    if kwargs["max_files"]:
        files_to_convert = files_to_convert[: kwargs["max_files"]]

    # Disable nested multiprocessing
    kwargs["disable_multiprocessing"] = True

    try:
        mp.set_start_method("spawn")  # Required for CUDA, forkserver doesn't work
    except RuntimeError:
        raise RuntimeError(
            "Set start method to spawn twice. This may be a temporary issue with the script. Please try running it again."
        )

    chunk_idx = kwargs["chunk_idx"]

    # Use GPU context manager for automatic setup/cleanup
    with GPUManager(chunk_idx) as gpu_manager:
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_manager, 7)

        # Override workers if specified
        if kwargs["workers"] is not None:
            workers = kwargs["workers"]

        # Set proper batch sizes and thread counts
        total_processes = max(1, min(len(files_to_convert), workers))
        kwargs["total_torch_threads"] = max(
            2, psutil.cpu_count(logical=False) // total_processes
        )
        kwargs.update(batch_sizes)

        logger.info(
            f"Converting {len(files_to_convert)} pdfs in chunk {kwargs['chunk_idx'] + 1}/{kwargs['num_chunks']} with {total_processes} processes and saving to {kwargs['output_dir']}"
        )
        task_args = [(f, kwargs) for f in files_to_convert]

        start_time = time.time()
        with mp.Pool(
            processes=total_processes,
            initializer=worker_init,
            maxtasksperchild=kwargs["max_tasks_per_worker"],
        ) as pool:
            pbar = tqdm(total=len(task_args), desc="Processing PDFs", unit="pdf")
            for page_count in pool.imap_unordered(process_single_pdf, task_args):
                pbar.update(1)
                total_pages += page_count
            pbar.close()

        total_time = time.time() - start_time
        print(
            f"Inferenced {total_pages} pages in {total_time:.2f} seconds, for a throughput of {total_pages / total_time:.2f} pages/sec for chunk {chunk_idx + 1}/{kwargs['num_chunks']}"
        )
