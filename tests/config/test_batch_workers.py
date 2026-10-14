import pytest
from click.testing import CliRunner

from marker.scripts.convert import convert_cli

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    "physical_cores, workers, expected_threads",
    [(None, 1, 4), (None, 2, 2), (8, 1, 8), (8, 2, 4), (1, 1, 2)],
)
def test_batch_thread_count(
    tmp_path, mocker, physical_cores, workers, expected_threads
):
    for index in range(workers):
        (tmp_path / f"sample-{index}.pdf").touch()
    mocker.patch("marker.scripts.convert.psutil.cpu_count", return_value=physical_cores)
    mocker.patch("marker.scripts.convert.mp.set_start_method")
    pool_factory = mocker.patch("marker.scripts.convert.mp.Pool")
    pool = pool_factory.return_value.__enter__.return_value
    pool.imap_unordered.return_value = [1] * workers

    result = CliRunner().invoke(
        convert_cli, [str(tmp_path), "--disable_ocr", "--workers", str(workers)]
    )

    assert result.exit_code == 0, result.exception
    assert pool_factory.call_args.kwargs["processes"] == workers
    task_args = pool.imap_unordered.call_args.args[1]
    assert len(task_args) == workers
    assert all(
        options["total_torch_threads"] == expected_threads for _, options in task_args
    )
