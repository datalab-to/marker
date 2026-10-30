from pathlib import Path

import pytest
from click.testing import CliRunner

from marker.scripts.convert import convert_cli

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    "directory_order", [("a.pdf", "b.pdf", "c.pdf"), ("c.pdf", "b.pdf", "a.pdf")]
)
@pytest.mark.parametrize(
    "chunk_idx, expected_files", [(0, ["a.pdf", "b.pdf"]), (1, ["c.pdf"])]
)
def test_shards_use_stable_file_order(
    tmp_path, mocker, directory_order, chunk_idx, expected_files
):
    for filename in directory_order:
        (tmp_path / filename).touch()
    mocker.patch(
        "marker.scripts.convert.os.listdir", return_value=list(directory_order)
    )
    mocker.patch("marker.scripts.convert.psutil.cpu_count", return_value=4)
    mocker.patch("marker.scripts.convert.mp.set_start_method")
    pool_factory = mocker.patch("marker.scripts.convert.mp.Pool")
    pool = pool_factory.return_value.__enter__.return_value
    pool.imap_unordered.return_value = [1] * len(expected_files)

    result = CliRunner().invoke(
        convert_cli,
        [
            str(tmp_path),
            "--disable_ocr",
            "--workers",
            "1",
            "--num_chunks",
            "2",
            "--chunk_idx",
            str(chunk_idx),
        ],
    )

    assert result.exit_code == 0, result.exception
    task_args = pool.imap_unordered.call_args.args[1]
    assert [Path(filename).name for filename, options in task_args] == expected_files
