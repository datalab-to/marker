import os

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
from marker.output import save_output
from marker.util import strings_to_classes

configure_logging()
logger = get_logger()


@click.command(cls=CustomClickPrinter, help="Convert a single PDF to markdown.")
@click.argument("fpath", type=str)
@ConfigParser.common_options
def convert_single_cli(fpath: str, **kwargs):
    models = create_model_dict()
    start = time.time()
    config_parser = ConfigParser(kwargs)

    converter_cls = config_parser.get_converter_cls()
    renderers = config_parser.get_renderers()                       # Get all requested renderers
    
    converter = converter_cls(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=renderers[0],                                      # Initialize converter with first renderer
        llm_service=config_parser.get_llm_service(),
    )
    document = None
    with converter.filepath_to_str(fpath) as temp_path:             # Build document only once
        document = converter.build_document(temp_path)
        converter.page_count = len(document.pages)
  
    out_folder = config_parser.get_output_folder(fpath)
    fname_base = config_parser.get_base_filename(fpath)
    for renderer_cls_str in renderers:                              # Render and save in all requested formats
        renderer_cls = strings_to_classes([renderer_cls_str])[0]
        renderer = converter.resolve_dependencies(renderer_cls)
        rendered = renderer(document)
        save_output(rendered, out_folder, fname_base)

    if len(renderers) > 1:
        logger.info(f"Saved {len(renderers)} format(s) to {out_folder}")
    else:
        logger.info(f"Saved output to {out_folder}")
    logger.info(f"Total time: {time.time() - start}")

if __name__ == "__main__":
    convert_single_cli()
