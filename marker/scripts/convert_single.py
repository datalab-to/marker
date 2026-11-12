import os
import sys

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Transformers uses .isin for a simple op, which is not supported on MPS
)

import time
import click

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output, text_from_rendered
from marker.settings import settings
from marker.skills import get_skill_markdown, install_skill

configure_logging()
logger = get_logger()


@click.command(
    cls=CustomClickPrinter,
    help="Convert a single PDF, image, or Office document to Markdown, JSON, Chunks, or HTML.",
)
@click.version_option(version="2.2.0", prog_name="marker-pdf")
@click.argument("fpath", type=str, required=False, default=None)
@click.option(
    "--stdout",
    is_flag=True,
    default=False,
    help="Write converted output directly to stdout instead of saving to a directory.",
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
def convert_single_cli(
    fpath: str | None,
    stdout: bool = False,
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
    if not fpath:
        raise click.UsageError("Missing argument 'FPATH'.")
    if not os.path.isfile(fpath):
        raise click.BadParameter(f"File not found: {fpath}", param_hint="'FPATH'")

    if kwargs.get("keep_server"):
        os.environ["SURYA_INFERENCE_KEEP_ALIVE"] = "1"
        try:
            from surya.settings import settings as surya_settings

            surya_settings.SURYA_INFERENCE_KEEP_ALIVE = True
        except Exception:
            pass

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
    load_indra = effective_mode not in ("vllm", "surya", "accurate")
    no_server = (
        bool(kwargs.get("disable_ocr"))
        or effective_mode == "no_ocr"
        or (uses_indra and effective_mode in ("fast", "indra"))
    )

    models = create_model_dict(load_indra=load_indra, no_server=no_server)
    start = time.time()
    config_parser = ConfigParser(kwargs)
    config_dict = config_parser.generate_config_dict()
    if stdout:
        config_dict["disable_tqdm"] = True

    converter_cls = config_parser.get_converter_cls()
    converter = converter_cls(
        config=config_dict,
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(fpath)

    if stdout:
        text, _, _ = text_from_rendered(rendered)
        try:
            sys.stdout.write(text)
            if not text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        except BrokenPipeError:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return

    out_folder = config_parser.get_output_folder(fpath)
    save_output(rendered, out_folder, config_parser.get_base_filename(fpath))

    output_format = config_dict.get("output_format", "markdown")
    logger.info(f"Saved {output_format} to {out_folder}")
    logger.info(f"Total time: {time.time() - start:.2f}s")
