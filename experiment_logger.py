"""Thin adapter that fans logging out to TensorBoard, Weights & Biases, or both.

The training loop only sees the methods on `ExperimentLogger` — it stays
backend-agnostic, so swapping or disabling a backend is a one-flag change.
"""
import os


class ExperimentLogger:
    def __init__(self, backends, run_dir, project=None, run_name=None, config=None, entity="meero-rd"):
        """
        backends: subset of {"tensorboard", "wandb"} — empty set is allowed (no-op logger).
        run_dir:  filesystem path for TB events (also passed to wandb.init dir=).
        project:  W&B project name.
        run_name: W&B run name (also used as the TB sub-directory implicitly via run_dir).
        config:   dict-like resolved hyperparameters to log against the run.
        entity:   W&B entity name.
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
                entity=entity,
                config=config or {},
                reinit="finish_previous",
            )
            # The training loop logs on two independent axes — per-iteration metrics keyed by
            # `global_step` and per-epoch metrics keyed by `epoch`. wandb enforces ONE monotonic
            # internal step across every log() call, so passing those axis values as `step=` breaks
            # the moment the two series interleave (and on resume, when `epoch` starts well above 0).
            # Instead we never pass an explicit step: wandb's auto-incrementing internal step stays
            # monotonic by construction, and each metric is bound to its real axis via define_metric.
            self._wb_axes_defined = set()
            self.wb.define_metric("global_step")
            self.wb.define_metric("epoch")

    def _wb_log(self, data, step, axis):
        """Log `data` to wandb against the logical `axis` ('global_step' or 'epoch'),
        without an explicit (monotonicity-constrained) wandb step."""
        for tag in data:
            if tag not in self._wb_axes_defined:
                self.wb.define_metric(tag, step_metric=axis)
                self._wb_axes_defined.add(tag)
        self.wb.log({**data, axis: step})

    # ------- scalars / images / text ----------------------------------------
    def add_scalar(self, tag, value, step, axis="global_step"):
        if self.tb is not None:
            self.tb.add_scalar(tag, value, step)
        if self.wb is not None:
            self._wb_log({tag: value}, step, axis)

    def add_scalars(self, mapping, step, axis="global_step"):
        for k, v in mapping.items():
            self.add_scalar(k, v, step, axis=axis)

    def add_image(self, tag, img_tensor, step, axis="global_step"):
        # img_tensor: CHW or HWC torch tensor / numpy array in [0,1] or [0,255].
        if self.tb is not None:
            self.tb.add_image(tag, img_tensor, step)
        if self.wb is not None:
            arr = img_tensor.detach().cpu().numpy() if hasattr(img_tensor, "detach") else img_tensor
            # wandb wants HWC uint8 or float. SummaryWriter accepts CHW; convert.
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.transpose(1, 2, 0)
            self._wb_log({tag: self._wandb.Image(arr)}, step, axis)

    def add_text(self, tag, text, step, axis="global_step"):
        if self.tb is not None:
            self.tb.add_text(tag, text, step)
        if self.wb is not None:
            # Logged as a single-cell wandb.Table so it shows up in the run UI.
            tbl = self._wandb.Table(columns=[tag], data=[[text]])
            self._wb_log({tag: tbl}, step, axis)

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
