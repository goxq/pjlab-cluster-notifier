from pjnotifier import TrainingNotifier
from pjnotifier.integrations import (
    HFTrainerNotificationCallback,
    HFTrainerNotificationConfig,
)
from pjnotifier.channels import FeishuChannel


training_notifier = TrainingNotifier(
    job_name="hf-trainer-demo",
    channels=FeishuChannel.from_env(),
    run_name="run-001",
    experiment_name="baseline",
    default_extra={"cluster": "gpu-a100"},
)


def attach_hf_trainer_notifications(trainer):
    trainer.add_callback(
        HFTrainerNotificationCallback(
            training_notifier,
            HFTrainerNotificationConfig(
                progress_every_n_steps=100,
                progress_min_interval_seconds=300,
            ),
        )
    )
    return trainer


@training_notifier.notify_on_failure(stage="train", include_traceback=True)
def run_training(trainer) -> None:
    attach_hf_trainer_notifications(trainer)
    trainer.train()
