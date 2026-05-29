"""Thin adapter that fans logging out to TensorBoard, Weights & Biases, or both.

The training loop only sees the methods on `ExperimentLogger` — it stays
backend-agnostic, so swapping or disabling a backend is a one-flag change.
"""
import os


class ExperimentLogger:
    def __init__(self, backends, run_dir, project=None, run_name=None, config=None):
        """
        backends: subset of {"tensorboard", "wandb"} — empty set is allowed (no-op logger).
        run_dir:  filesystem path for TB events (also passed to wandb.init dir=).
        project:  W&B project name.
        run_name: W&B run name (also used as the TB sub-directory implicitly via run_dir).
        config:   dict-like resolved hyperparameters to log against the run.
        """
        backends = set(backends or [])
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.backends = backends
        self.tb = None
        self.wb = None

        if "tensorboard" in backends:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=run_dir)

        if "wandb" in backends:
            import wandb
            self._wandb = wandb
            self.wb = wandb.init(
                project=project,
                name=run_name,
                dir=run_dir,
                config=config or {},
                reinit="finish_previous",
            )

    # ------- scalars / images / text ----------------------------------------
    def add_scalar(self, tag, value, step):
        if self.tb is not None:
            self.tb.add_scalar(tag, value, step)
        if self.wb is not None:
            self.wb.log({tag: value}, step=step)

    def add_scalars(self, mapping, step):
        for k, v in mapping.items():
            self.add_scalar(k, v, step)

    def add_image(self, tag, img_tensor, step):
        # img_tensor: CHW or HWC torch tensor / numpy array in [0,1] or [0,255].
        if self.tb is not None:
            self.tb.add_image(tag, img_tensor, step)
        if self.wb is not None:
            arr = img_tensor.detach().cpu().numpy() if hasattr(img_tensor, "detach") else img_tensor
            # wandb wants HWC uint8 or float. SummaryWriter accepts CHW; convert.
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.transpose(1, 2, 0)
            self.wb.log({tag: self._wandb.Image(arr)}, step=step)

    def add_text(self, tag, text, step):
        if self.tb is not None:
            self.tb.add_text(tag, text, step)
        if self.wb is not None:
            # Logged as a single-cell wandb.Table so it shows up in the run UI.
            tbl = self._wandb.Table(columns=[tag], data=[[text]])
            self.wb.log({tag: tbl}, step=step)

    # ------- lifecycle ------------------------------------------------------
    def flush(self):
        if self.tb is not None:
            self.tb.flush()

    def close(self):
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self.wb is not None:
            self.wb.finish()


def parse_backends(flag_value):
    """Map the --logger CLI value to a backend set. Unknown values raise."""
    if not flag_value or flag_value == "none":
        return set()
    if flag_value == "both":
        return {"tensorboard", "wandb"}
    if flag_value in ("tensorboard", "wandb"):
        return {flag_value}
    raise ValueError(
        "Unknown --logger value: {!r}. Expected one of: tensorboard, wandb, both, none.".format(flag_value)
    )
