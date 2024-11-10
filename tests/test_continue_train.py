from tests import run_cli


def test_continue_train(tmp_path):
    command_train = f"python tests/utils/train_mnist.py --coqpit.output_path {tmp_path}"
    run_cli(command_train)

    continue_path = max(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime)
    number_of_checkpoints = len(list(continue_path.glob("*.pth")))

    # Continue training from the best model
    command_continue = f"python tests/utils/train_mnist.py --continue_path {continue_path} --coqpit.run_eval_steps=1"
    run_cli(command_continue)

    assert number_of_checkpoints < len(list(continue_path.glob("*.pth")))

    # Continue training from the last checkpoint
    for best in continue_path.glob("best_model*"):
        best.unlink()
    run_cli(command_continue)

    # Continue training from a specific checkpoint
    restore_path = continue_path / "checkpoint_5.pth"
    command_continue = (
        f"python tests/utils/train_mnist.py --restore_path {restore_path} --coqpit.output_path {tmp_path}"
    )
    run_cli(command_continue)
