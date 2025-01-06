import sys

from tests.utils.train_mnist import main as train_mnist


def test_continue_train(tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", [sys.argv[0], "--coqpit.output_path", str(tmp_path)])
        train_mnist()

    continue_path = max(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime)
    number_of_checkpoints = len(list(continue_path.glob("*.pth")))

    # Continue training from the best model
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", [sys.argv[0], "--continue_path", str(continue_path), "--coqpit.run_eval_steps", "1"])
        train_mnist()

    assert number_of_checkpoints < len(list(continue_path.glob("*.pth")))

    # Continue training from the last checkpoint
    for best in continue_path.glob("best_model*"):
        best.unlink()

    # Continue training from a specific checkpoint
    restore_path = continue_path / "checkpoint_5.pth"
    with monkeypatch.context() as m:
        m.setattr(
            sys, "argv", [sys.argv[0], "--restore_path", str(restore_path), "--coqpit.output_path", str(tmp_path)]
        )
        train_mnist()
