# pjlab-cluster-notifier

A lightweight Python package for sending to workplace chat apps (e.g. Feishu) runtime
notifications from training jobs and related workflows.

## Usage

Install from GitHub:

```bash
uv pip install "git+https://github.com/goxq/pjlab-cluster-notifier.git"
```

Install the local checkout in editable mode:

```bash
uv pip install -e .
```

Basic training loop:

```python
from pjnotifier import TrainingNotifier
from pjnotifier.channels import FeishuChannel

training_notifier = TrainingNotifier(
    job_name="demo-training",
    channels=FeishuChannel.from_env(),
)

training_notifier.train_started(total_steps=1000)
training_notifier.train_progress(step=100, total_steps=1000, loss=0.42)
training_notifier.eval_metrics(metrics={"eval_loss": 0.35, "accuracy": 0.91})
training_notifier.train_finished(best_metric=("accuracy", 0.91))
```

Hugging Face Trainer:

```python
from pjnotifier import TrainingNotifier
from pjnotifier.channels import FeishuChannel
from pjnotifier.integrations import HFTrainerNotificationCallback

training_notifier = TrainingNotifier(
    job_name="hf-training",
    channels=FeishuChannel.from_env(),
)

trainer.add_callback(HFTrainerNotificationCallback(training_notifier))
trainer.train()
```

Required environment variables for the Feishu channel:

- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`
- `FEISHU_RECEIVE_ID`

Optional environment variables:

- `FEISHU_RECEIVE_ID_TYPE` default: `open_id`
- `FEISHU_BASE_URL`
- `FEISHU_TIMEOUT`
- `FEISHU_MESSAGE_STYLE` default: `interactive` (`interactive` uses a Feishu card, `post` uses a simpler rich-text block, `text` keeps plain text)

## Development

```bash
uv sync
```
