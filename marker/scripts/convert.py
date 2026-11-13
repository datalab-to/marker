import atexit
import os
import sys
import time
from typing import List

import psutil

from marker.utils.batch import get_worker_count

# Ensure threads don't contend
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "2"  # Avoid OpenMP issues with multiprocessing
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import gc
import math
import multiprocessing as mp
import traceback

import click
from tqdm import tqdm

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import output_exists, save_output
from marker.settings import settings
from marker.skills import get_skill_markdown, install_skill

configure_logging()
logger = get_logger()

SUPPORTED_FILE_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
    ".pptx",
    ".docx",
    ".xlsx",
    ".html",
    ".htm",
    ".epub",
}


def discover_input_files(in_path: str, recursive: bool = False) -> List[str]:
    """Discover supported document files from a file path or directory."""
    in_path = os.path.abspath(in_path)
    if os.path.isfile(in_path):
        return [in_path]
    if not os.path.isdir(in_path):
        raise FileNotFoundError(f"Input path does not exist: {in_path}")

    discovered: List[str] = []
    if recursive:
        for root, dirs, filenames in os.walk(in_path):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_FILE_EXTENSIONS:
                    full = os.path.join(root, fname)
                    if os.path.isfile(full):
                        discovered.append(full)
    else:
        for fname in sorted(os.listdir(in_path)):
            if fname.startswith("."):
                continue
            full = os.path.join(in_path, fname)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_FILE_EXTENSIONS:
                discovered.append(full)
    return discovered


def worker_init(
    model_device: str | None = None,
    load_indra: bool = True,
    no_server: bool = False,
):
    # The VLM predictors attach to the server the parent process spawned
    # (via SURYA_INFERENCE_URL). Only the small ocr error model loads locally.
    model_dict = create_model_dict(
        device=model_device,
        load_indra=load_indra,
        no_server=no_server,
    )

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
    cli_options = dict(cli_options)
    torch_threads = cli_options.pop("total_torch_threads", 2)
    base_in_folder = cli_options.pop("_base_in_folder", None)
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)

    config_parser = ConfigParser(cli_options)

    out_folder = config_parser.get_output_folder(
        fpath, base_in_folder=base_in_folder
    )
    base_name = config_parser.get_base_filename(fpath)
    if cli_options.get("skip_existing") and output_exists(out_folder, base_name):
        return (0, "skipped")

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
        out_folder = config_parser.get_output_folder(
            fpath, base_in_folder=base_in_folder
        )
        save_output(rendered, out_folder, base_name)
        page_count = converter.page_count or 0

        if cli_options.get("debug_print"):
            logger.debug(f"Converted {fpath} ({page_count} pages)")
        del rendered
        del converter
        return (page_count, "ok")
    except Exception as e:
        logger.error(f"Error converting {fpath}: {e}")
        traceback.print_exc()
        return (0, "error")
    finally:
        gc.collect()


@click.command(
    cls=CustomClickPrinter,
    help="Convert a folder of documents (or a single file) to Markdown, JSON, Chunks, or HTML.",
)
@click.version_option(version="2.2.1", prog_name="marker-pdf")
@click.argument("in_folder", type=str, required=False, default=None)
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
    "--max_files", type=int, default=None, help="Maximum number of files to convert."
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    default=False,
    help="Recursively search subdirectories for documents.",
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
    help="Number of worker processes to use. Set automatically by default, but can be overridden.",
)
@click.option(
    "--skill",
    is_flag=True,
    default=False,
    help="Print the Marker AI agent skill (SKILL.md) to stdout and exit.",
)
@click.option(
    "--install_skill",
    type=str,
    default=None,
    help="Install the Marker SKILL.md into the given directory (e.g. .agents/skills) and exit.",
)
@ConfigParser.common_options
def convert_cli(
    in_folder: str | None,
    skill: bool = False,
    install_skill_dir: str | None = None,
    **kwargs,
):
    install_dir = kwargs.pop("install_skill", None) or install_skill_dir
    if skill:
        sys.stdout.write(get_skill_markdown())
        return
    if install_dir:
        dest = install_skill(install_dir)
        click.echo(f"Installed Marker skill to {dest}")
        return
    if not in_folder:
        raise click.UsageError("Missing argument 'IN_FOLDER'.")

    total_pages = 0
    in_folder = os.path.abspath(in_folder)
    recursive = bool(kwargs.pop("recursive", False))
    user_disabled_mp = bool(kwargs.get("disable_multiprocessing", False))

    files = discover_input_files(in_folder, recursive=recursive)
    if not files:
        logger.info(f"No supported documents found in {in_folder}.")
        return

    base_in_folder = (
        in_folder if (recursive and os.path.isdir(in_folder)) else None
    )

    # Handle chunks if we're processing in parallel across nodes
    num_chunks = max(1, int(kwargs["num_chunks"]))
    chunk_idx = int(kwargs["chunk_idx"])
    chunk_size = math.ceil(len(files) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size
    files_to_convert = files[start_idx:end_idx]

    # Limit files converted if needed
    if kwargs["max_files"]:
        files_to_convert = files_to_convert[: kwargs["max_files"]]

    # Pre-filter already-converted files when --skip_existing is set so we
    # don't spin up workers or an inference server for already-finished work.
    skipped_pre = 0
    if kwargs.get("skip_existing"):
        preview_parser = ConfigParser(kwargs)
        remaining = []
        for fpath in files_to_convert:
            out_dir = preview_parser.get_output_folder(
                fpath, base_in_folder=base_in_folder
            )
            base_name = preview_parser.get_base_filename(fpath)
            if output_exists(out_dir, base_name):
                skipped_pre += 1
            else:
                remaining.append(fpath)
        if skipped_pre:
            logger.info(f"Skipping {skipped_pre} already-converted file(s).")
        files_to_convert = remaining

    if not files_to_convert:
        logger.info("No files left to convert in this chunk.")
        return

    if kwargs.get("keep_server"):
        os.environ["SURYA_INFERENCE_KEEP_ALIVE"] = "1"
        try:
            from surya.settings import settings as surya_settings

            surya_settings.SURYA_INFERENCE_KEEP_ALIVE = True
        except Exception:
            pass

    # Disable nested multiprocessing inside workers
    kwargs["disable_multiprocessing"] = True
    kwargs["_base_in_folder"] = base_in_folder

    raw_mode = kwargs.get("mode")
    effective_mode = (
        None if raw_mode in (None, "auto") else raw_mode
    ) or ("balanced" if settings.TORCH_DEVICE_MODEL == "cuda" else "fast")
    if kwargs.get("disable_ocr"):
        effective_mode = "no_ocr"

    ocr_backend = kwargs.get("ocr_backend") or "auto"
    uses_indra = ocr_backend == "indra" or (
        ocr_backend == "auto" and effective_mode in ("fast", "indra")
    )
    no_server = (
        bool(kwargs.get("disable_ocr"))
        or effective_mode == "no_ocr"
        or (uses_indra and effective_mode in ("fast", "indra"))
    )
    load_indra = effective_mode not in ("vllm", "surya", "accurate")

    if no_server:
        # Pure CPU / ONNX mode: no VLM server, pool sized by CPU cores.
        workers = (
            1
            if user_disabled_mp
            else (kwargs["workers"] or get_worker_count(no_server=True))
        )
        total_processes = max(1, min(len(files_to_convert), workers))
    else:
        # Spawn (or attach to) the shared inference server from the parent
        # process so it outlives worker recycling and is cleaned up when this
        # process exits. Workers attach to it via SURYA_INFERENCE_URL.
        from surya.inference import SuryaInferenceManager

        manager = SuryaInferenceManager(lazy=False)
        server_handle = manager.backend.start()  # Idempotent, returns handle
        os.environ["SURYA_INFERENCE_URL"] = server_handle.base_url

        workers = (
            1 if user_disabled_mp else (kwargs["workers"] or get_worker_count())
        )
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

    # The VLM runs on the server; workers only hold the ONNX models.
    # Keep them off the GPU in multi-worker mode - the server owns the VRAM.
    model_device = None
    if total_processes > 1 and settings.TORCH_DEVICE_MODEL == "cuda":
        model_device = "cpu"

    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 2
    kwargs["total_torch_threads"] = max(2, cpu_cores // total_processes)

    logger.info(
        f"Converting {len(files_to_convert)} document(s) in chunk {chunk_idx + 1}/{num_chunks} "
        f"(mode={effective_mode}) with {total_processes} process(es) -> {kwargs['output_dir']}"
    )
    task_args = [(f, kwargs) for f in files_to_convert]

    start_time = time.time()
    converted_files = 0
    failed_files = 0

    if total_processes == 1:
        # Fast path: single file or --workers 1 / --disable_multiprocessing
        # runs in-process without multiprocessing spawn overhead.
        worker_init(model_device, load_indra, no_server)
        try:
            pbar = tqdm(total=len(task_args), desc="Processing documents", unit="doc")
            for item in task_args:
                res = process_single_pdf(item)
                pages, status = res if isinstance(res, tuple) else (int(res or 0), "ok")
                pbar.update(1)
                total_pages += pages
                if status == "ok":
                    converted_files += 1
                elif status == "error":
                    failed_files += 1
            pbar.close()
        finally:
            worker_exit()
    else:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        with mp.Pool(
            processes=total_processes,
            initializer=worker_init,
            initargs=(model_device, load_indra, no_server),
            maxtasksperchild=kwargs["max_tasks_per_worker"],
        ) as pool:
            pbar = tqdm(total=len(task_args), desc="Processing documents", unit="doc")
            for res in pool.imap_unordered(process_single_pdf, task_args):
                pages, status = res if isinstance(res, tuple) else (int(res or 0), "ok")
                pbar.update(1)
                total_pages += pages
                if status == "ok":
                    converted_files += 1
                elif status == "error":
                    failed_files += 1
            pbar.close()

    total_time = max(1e-6, time.time() - start_time)
    summary_msg = (
        f"Converted {converted_files} file(s) ({total_pages} pages) in {total_time:.2f}s "
        f"({total_pages / total_time:.2f} pages/sec) for chunk {chunk_idx + 1}/{num_chunks}"
    )
    if skipped_pre:
        summary_msg += f" [{skipped_pre} skipped]"
    if failed_files:
        summary_msg += f" [{failed_files} failed]"
    print(summary_msg)
