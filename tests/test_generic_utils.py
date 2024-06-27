from pathlib import Path

from trainer.generic_utils import remove_experiment_folder


def test_remove_experiment_folder():
    output_dir = Path("output")
    run_dir = output_dir / "run"
    run_dir.mkdir(exist_ok=True, parents=True)

    remove_experiment_folder(run_dir)
    assert not run_dir.is_dir()

    run_dir.mkdir(exist_ok=True, parents=True)
    checkpoint = run_dir / "checkpoint.pth"
    checkpoint.touch(exist_ok=False)
    remove_experiment_folder(run_dir)
    assert checkpoint.is_file()

    remove_experiment_folder(str(run_dir) + "/")
    assert checkpoint.is_file()

    checkpoint.unlink()
    run_dir.rmdir()
