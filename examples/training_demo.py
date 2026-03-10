import time

from pjnotifier import TrainingNotifier
from pjnotifier.channels import FeishuChannel


training_notifier = TrainingNotifier(
    job_name="demo-train",
    channels=FeishuChannel.from_env(),
    run_name="run-001",
    experiment_name="baseline",
    default_extra={"cluster": "gpu-a100"},
)


@training_notifier.notify_on_failure(stage="train")
def train() -> None:
    training_notifier.train_started(
        total_steps=3,
        learning_rate=1e-4,
        model="demo-model",
        dataset="demo-dataset",
    )

    for step in range(1, 4):
        loss = 1.0 / step
        training_notifier.train_progress(
            step=step,
            total_steps=3,
            epoch=step / 3,
            loss=loss,
            learning_rate=1e-4,
            throughput=32.5,
            eta_seconds=3 - step,
        )
        time.sleep(1)

        if step == 2:
            training_notifier.eval_started(split="validation", step=step)
            training_notifier.eval_metrics(
                split="validation",
                step=step,
                metrics={"eval_loss": 0.23, "accuracy": 0.91},
            )

    training_notifier.train_finished(best_metric=("accuracy", 0.91))


if __name__ == "__main__":
    train()
