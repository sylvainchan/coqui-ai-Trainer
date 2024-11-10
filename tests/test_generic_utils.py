from trainer.generic_utils import remove_experiment_folder


def test_remove_experiment_folder(tmp_path):
    run_dir = tmp_path / "run"
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
