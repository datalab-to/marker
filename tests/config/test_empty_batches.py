import pytest
from click.testing import CliRunner

from marker.scripts.convert import convert_cli

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("empty_shard", [False, True])
@pytest.mark.parametrize("disable_ocr", [False, True])
def test_empty_batch_skips_initialization(tmp_path, mocker, empty_shard, disable_ocr):
    arguments = [str(tmp_path)]
    if empty_shard:
        (tmp_path / "sample.pdf").touch()
        arguments.extend(["--num_chunks", "2", "--chunk_idx", "1"])
    if disable_ocr:
        arguments.append("--disable_ocr")

    start_method = mocker.patch("marker.scripts.convert.mp.set_start_method")
    pool = mocker.patch(
        "marker.scripts.convert.mp.Pool",
        side_effect=RuntimeError("worker pool unavailable"),
    )
    backend = mocker.patch(
        "surya.inference.SuryaInferenceManager",
        side_effect=RuntimeError("inference backend unavailable"),
    )

    result = CliRunner().invoke(convert_cli, arguments)

    assert result.exit_code == 0, result.exception
    start_method.assert_not_called()
    backend.assert_not_called()
    pool.assert_not_called()


def test_nonempty_batch_still_creates_workers(tmp_path, mocker):
    (tmp_path / "sample.pdf").touch()
    mocker.patch("marker.scripts.convert.psutil.cpu_count", return_value=4)
    mocker.patch("marker.scripts.convert.mp.set_start_method")
    pool_factory = mocker.patch("marker.scripts.convert.mp.Pool")
    pool = pool_factory.return_value.__enter__.return_value
    pool.imap_unordered.return_value = [1]

    result = CliRunner().invoke(
        convert_cli, [str(tmp_path), "--disable_ocr", "--workers", "1"]
    )

    assert result.exit_code == 0, result.exception
    pool_factory.assert_called_once()
    assert pool_factory.call_args.kwargs["processes"] == 1
    task_args = pool.imap_unordered.call_args.args[1]
    assert len(task_args) == 1
    assert task_args[0][0] == str(tmp_path / "sample.pdf")
