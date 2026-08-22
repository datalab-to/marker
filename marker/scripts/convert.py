import atexit
import os
import time

import psutil
import torch

from marker.utils.batch import get_worker_count

# Let unsupported MPS ops fall back to CPU (Apple Silicon); matches the other
# entrypoints. Workers/servers spawned here inherit this.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Ensure threads don't contend
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "2"  # Avoid OpenMP issues with multiprocessing
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
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

configure_logging()
logger = get_logger()


def worker_init(model_device: str | None = None):
    # The VLM predictors attach to the server the parent process spawned
    # (via SURYA_INFERENCE_URL). Only the small ocr error model loads locally.
    model_dict = create_model_dict(device=model_device)

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
    if cli_options.get("skip_existing") and output_exists(
        out_folder, base_name, cli_options.get("output_format")
    ):
        return page_count

    converter_cls = config_parser.get_converter_cls()
    config_dict = config_parser.generate_config_dict()
    config_dict["disable_tqdm"] = True
    # This is a worker process in a multiprocessing pool. pdftext runs a
    # ProcessPoolExecutor of its own when workers > 1 (pypdfium2 is not
    # thread-safe, so it forks per-page work); nesting that inside our worker
    # pool deadlocks. Force in-process pdftext (workers<=1) so each worker holds
    # exactly one pdftext instance. Throughput comes from the pool's breadth.
    config_dict["pdftext_workers"] = 1

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
@click.option(
    "--chunk_idx",
    type=int,
    default=0,
    help="This node's shard index (multi-node: give each node a distinct chunk_idx).",
)
@click.option(
    "--num_chunks",
    type=int,
    default=1,
    help="Shard the file list into this many chunks (multi-node batch runs: "
    "one marker invocation per node, each with its own inference server).",
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
    default=200,
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

    no_server = bool(kwargs.get("disable_ocr"))
    if no_server:
        # Pure text-layer mode: no VLM at all, so no server and the pool is
        # sized purely by CPU cores.
        workers = kwargs["workers"] or get_worker_count(no_server=True)
        total_processes = max(1, min(len(files_to_convert), workers))
    else:
        # Spawn (or attach to) the shared inference server from the parent
        # process so it outlives worker recycling and is cleaned up when this
        # process exits. Workers attach to it via SURYA_INFERENCE_URL.
        from surya.inference import SuryaInferenceManager

        manager = SuryaInferenceManager(lazy=False)
        server_handle = manager.backend.start()  # Idempotent, returns handle
        os.environ["SURYA_INFERENCE_URL"] = server_handle.base_url

        workers = kwargs["workers"] or get_worker_count()
        total_processes = max(1, min(len(files_to_convert), workers))

        # Budget aggregate client concurrency to ~1.5x the server's capacity:
        # each worker's client otherwise fans out to the FULL capacity on its
        # own, so N workers would queue N-times-capacity requests. Respect an
        # explicit SURYA_INFERENCE_PARALLEL from the user.
        if "SURYA_INFERENCE_PARALLEL" not in os.environ:
            capacity = manager.capacity()
            per_worker = max(1, math.ceil(capacity * 1.5 / total_processes))
            os.environ["SURYA_INFERENCE_PARALLEL"] = str(per_worker)
            logger.info(
                f"Server capacity {capacity}; using {per_worker} concurrent "
                f"requests per worker across {total_processes} workers"
            )

    # The VLM runs on the server; workers only hold the small ocr error model.
    # Keep it off the GPU in multi-worker mode - the server owns the VRAM.
    model_device = None
    if total_processes > 1 and torch.cuda.is_available():
        model_device = "cpu"

    kwargs["total_torch_threads"] = max(
        2, psutil.cpu_count(logical=False) // total_processes
    )

    logger.info(
        f"Converting {len(files_to_convert)} pdfs in chunk {kwargs['chunk_idx'] + 1}/{kwargs['num_chunks']} with {total_processes} processes and saving to {kwargs['output_dir']}"
    )
    task_args = [(f, kwargs) for f in files_to_convert]

    start_time = time.time()
    with mp.Pool(
        processes=total_processes,
        initializer=worker_init,
        initargs=(model_device,),
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
