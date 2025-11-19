import os
import shlex
import subprocess

import click

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter


def build_cli_args(kwargs):
    """Convert kwargs dictionary to a list of CLI argument strings."""
    args = []
    exclude_params = {
        "chunk_idx",
        "num_chunks",
        "workers",
        "in_folder",
        "output_dir",
    }  # Handled by chunk_convert.sh

    for key, value in kwargs.items():
        if key in exclude_params or value is None:
            continue

        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(str(value))

    return args


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
def chunk_convert_cli(in_folder: str, **kwargs):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(cur_dir, "chunk_convert.sh")

    cli_args = build_cli_args(kwargs)

    # Get output_dir with default fallback
    output_dir = kwargs.get("output_dir", "")

    # Construct the command
    escaped_args = " ".join(shlex.quote(arg) for arg in cli_args)
    cmd = f"{script_path} {shlex.quote(in_folder)} {shlex.quote(output_dir)} {escaped_args}"

    # Execute the shell script
    subprocess.run(cmd, shell=True, check=True)
