import time
import logging
from typing import Any, Optional

import torch
import lightning as L
from lightning.fabric.utilities.throughput import measure_flops
import numpy as np
try:
    import anndata as ad
except Exception:
    ad = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RegFreezeCallback(L.Callback):
    def __init__(self, warmup_steps: int, freeze_after_warmup: bool = True):
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.freeze_after_warmup = bool(freeze_after_warmup)
        self._frozen = False

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx):
        if not self.freeze_after_warmup or self._frozen:
            return
        if trainer.global_step >= self.warmup_steps and getattr(pl_module, "reg_enabled", False):
            # Freeze only the projector by default; LN/Dropout have no params/low risk
            reg_proj = getattr(pl_module, "reg_proj", None)
            if reg_proj is not None:
                for p in reg_proj.parameters():
                    p.requires_grad = False
            self._frozen = True
            if trainer.logger is not None:
                trainer.logger.log_metrics({"se_reg/frozen": 1, "se_reg/freeze_step": trainer.global_step}, step=trainer.global_step)


class LogLR(L.Callback):
    def __init__(self, interval=10):
        super().__init__()
        self.interval = interval

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *args,
    ) -> None:
        if trainer.global_rank == 0:
            if trainer.global_step % self.interval == 0 and trainer.logger is not None:
                trainer.logger.log_metrics(
                    {"trainer/learning_rate": pl_module.lr_schedulers().get_last_lr()[0]},
                    step=trainer.global_step,
                )


class PerfProfilerCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.batch_start_time = None
        self.batch_times = []
        self.iterations_count = 0
        self.last_ipm_time = None
        self.ipm_history = []

    def on_train_batch_start(self, trainer: L.Trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()

        # Calculate batch time
        if self.batch_start_time:
            batch_time = current_time - self.batch_start_time
            self.batch_times.append(batch_time)

        # Track iterations per minute
        self.iterations_count += 1
        if self.last_ipm_time is None:
            self.last_ipm_time = current_time

        time_diff = current_time - self.last_ipm_time
        if time_diff >= 60:
            ipm = (self.iterations_count / time_diff) * 60
            self.ipm_history.append(ipm)
            trainer.logger.log_metrics({"perf/ipm": ipm}, step=trainer.global_step)
            # Reset counters
            self.iterations_count = 0
            self.last_ipm_time = current_time


class ProfilerCallback(L.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.batch_start_time = None
        self.batch_times = []
        self.iterations_count = 0
        self.last_ipm_time = None
        self.ipm_history = []
        self.cfg = cfg

        self.profile_steps = cfg.experiment.profile.profile_steps

    def on_train_batch_start(self, trainer: L.Trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        if batch_idx == self.profile_steps[0]:
            logging.info(f"Starting NSys profiling at step {batch_idx}")
            torch.cuda.nvtx.range_push("VCIProfiledSection")

    def on_train_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()

        # Calculate batch time
        if self.batch_start_time:
            batch_time = current_time - self.batch_start_time
            self.batch_times.append(batch_time)

        # Track iterations per minute
        self.iterations_count += 1
        if self.last_ipm_time is None:
            self.last_ipm_time = current_time

        time_diff = current_time - self.last_ipm_time
        if time_diff >= 60:
            ipm = (self.iterations_count / time_diff) * 60
            self.ipm_history.append(ipm)
            trainer.logger.log_metrics({"perf/ipm": ipm}, step=trainer.global_step)
            # Reset counters
            self.iterations_count = 0
            self.last_ipm_time = current_time

        if batch_idx == self.profile_steps[1]:
            logging.info(f"Stopping NSys profiling at step {batch_idx}")
            torch.cuda.nvtx.range_pop()


class ResumeCallback(L.Callback):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

    def on_train_start(self, trainer, pl_module):
        if self._cfg.optimizer.get("reset_lr_on_restart", False):
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    original_lr = param_group.get("lr", None)
                    param_group["lr"] = self._cfg.optimizer.max_lr
                    logging.info(f"Reset learning rate from {original_lr} to {param_group['lr']}")


class EMACallback(L.Callback):
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.beta = decay
        self.velocity = {}

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer):
        # Check if EMA is enabled via the config flag.
        if pl_module.cfg.model.get("ema", False):
            with torch.no_grad():
                for param in pl_module.parameters():
                    if param.grad is None:
                        continue

                    param_id = id(param)
                    if param_id not in self.velocity:
                        self.velocity[param_id] = torch.zeros_like(param.grad)

                    self.velocity[param_id] = self.beta * self.velocity[param_id] + (1 - self.beta) * param.grad
                    param.grad = self.velocity[param_id].clone()


class CumulativeFLOPSCallback(L.Callback):
    """
    PyTorch Lightning callback to track cumulative FLOPS during SE training.

    - Measures FLOPs once on the first training batch using `measure_flops`.
    - Tracks cumulative FLOPs and logs at validation frequency.
    - Logs cumulative_flops to trainer loggers (e.g., W&B, CSV) at validation cadence.

    Args:
        use_backward: If True, include backward pass FLOPs in the measurement.
    """

    def __init__(
        self,
        *,
        use_backward: bool = False,
    ) -> None:
        super().__init__()
        self.use_backward = use_backward

        self._flops_per_batch: Optional[int] = None
        self._measured: bool = False
        self._cumulative_flops: int = 0
        self._batch_count: int = 0
        self._last_batch_start_time: Optional[float] = None

    def _trainstep_forward_backward(self, model: L.LightningModule, batch: Any) -> torch.Tensor:
        """Encapsulate calling StateEmbeddingModel.training_step and backward.

        This intentionally targets StateEmbeddingModel's signature and
        performs both forward and backward to capture full FLOPs.

        !!WARNING!!
        This has only been tested with StateEmbeddingModel. Behavior with any other model has not been verified.
        """
        model.zero_grad(set_to_none=True)
        loss: torch.Tensor = model.training_step(batch, 0)  # type: ignore
        if self.use_backward:
            loss.backward()
        return loss

    def _measure_flops_once(self, trainer: L.Trainer, pl_module: Any, batch: Any) -> None:
        if self._measured:
            return

        model = pl_module
        def forward_fn():
            return self._trainstep_forward_backward(model, batch)

        # Be resilient to OOM during FLOPs measurement (can spike memory use with SDPA)
        try:
            self._flops_per_batch = int(measure_flops(model, forward_fn=forward_fn))
            logger.info(f"CumulativeFLOPSCallback: Measured FLOPs per batch: {self._flops_per_batch}")
        except RuntimeError as e:
            # Commonly torch.OutOfMemoryError (subclass of RuntimeError); skip FLOPs measurement
            logger.warning(f"CumulativeFLOPSCallback: Skipping FLOPs measurement due to error: {e}")
            self._flops_per_batch = None

        model.zero_grad(set_to_none=True)
        self._measured = True

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: Any, batch: dict, batch_idx: int) -> None:
        if not self._measured and batch_idx == 0 and trainer.current_epoch == 0:
            self._measure_flops_once(trainer, pl_module, batch)
        # mark start time to compute step time
        self._last_batch_start_time = time.time()

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: Any, outputs: Any, batch: dict, batch_idx: int) -> None:
        if self._flops_per_batch is None:
            return

        self._batch_count += 1
        self._cumulative_flops += self._flops_per_batch

        # Log cumulative FLOPs after every training batch
        pl_module.log(
            "cumulative_flops",
            float(self._cumulative_flops),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        # Also compute MFU if timing is available
        try:
            if self._last_batch_start_time is not None:
                step_time = max(1e-6, time.time() - self._last_batch_start_time)
                # Read peak TFLOPs from config if present; default to 989 TFLOPs for H200 (bf16)
                peak_tflops = float(getattr(getattr(pl_module, "cfg", {}).experiment, "peak_tflops_bf16", 989))
                peak_flops_per_gpu = peak_tflops * 1e12
                mfu = float(self._flops_per_batch) / (peak_flops_per_gpu * step_time)
                # Clamp to [0,1] for sanity
                mfu = max(0.0, min(1.0, mfu))
                trainer.logger.log_metrics({"perf/mfu": mfu}, step=trainer.global_step)
        except Exception as e:
            logger.debug(f"MFU logging skipped due to error: {e}")

    def on_validation_start(self, trainer: L.Trainer, pl_module: Any) -> None:
        if self._flops_per_batch is None:
            return

        # Log cumulative FLOPs at validation frequency for W&B panel alignment
        pl_module.log(
            "cumulative_flops_val_sync",
            float(self._cumulative_flops),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )


class ProbeEvalCallback(L.Callback):
    """
    Periodically runs SE probe metrics on a small dev h5ad and logs to W&B:
      - probes/pearson_delta
      - probes/pert_rank

    Cadence controlled by:
      cfg.validations.diff_exp.eval_interval_multiple
      cfg.validations.perturbation.eval_interval_multiple
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._val_calls = 0
        self._dev_adata = None
        self._dev_name = None
        self._enabled = False
        try:
            self._de_conf = getattr(cfg, "validations").diff_exp if hasattr(getattr(cfg, "validations"), "diff_exp") else None
            self._pert_conf = getattr(cfg, "validations").perturbation if hasattr(getattr(cfg, "validations"), "perturbation") else None
            self._enabled = bool(getattr(self._de_conf, "enable", False) or getattr(self._pert_conf, "enable", False))
        except Exception:
            self._de_conf = None
            self._pert_conf = None
            self._enabled = False

    def _ensure_dev(self):
        if not self._enabled:
            return False
        try:
            if self._dev_adata is None:
                if ad is None:
                    return False
                # Prefer perturbation dataset if specified; else fall back to diff_exp
                if self._pert_conf and getattr(self._pert_conf, "dataset", None):
                    path = self._pert_conf.dataset
                    name = getattr(self._pert_conf, "dataset_name", "dev_probe")
                elif self._de_conf and getattr(self._de_conf, "dataset", None):
                    path = self._de_conf.dataset
                    name = getattr(self._de_conf, "dataset_name", "dev_probe")
                else:
                    return False
                self._dev_adata = ad.read_h5ad(path)
                self._dev_name = name
            return True
        except Exception as e:
            logger.debug(f"ProbeEvalCallback: failed to load dev adata: {e}")
            return False

    @torch.no_grad()
    def _compute_batch_preds(self, pl_module: L.LightningModule, batch: Any):
        """
        Compute predictions for the collated batch without advancing the LR schedulers.
        Returns (pred_matrix, true_matrix, labels_array, gene_index_array) aligned on a per-batch gene union.
        """
        # Unpack batch following VCIDatasetSentenceCollator ordering
        bs_indices = batch[0].to(pl_module.device, non_blocking=True)  # [B, pad_len]
        X_indices = batch[1].to(pl_module.device, non_blocking=True)   # [B, task_num] (global gene indices)
        Y_targets = batch[2].to(pl_module.device, non_blocking=True)   # [B, task_num]
        idxs = batch[3].detach().cpu().numpy()                          # [B]
        batch_weights = batch[4]  # unused
        masks = batch[5].to(pl_module.device, non_blocking=True)       # [B, pad_len] bool
        batch_sentences_counts = batch[7] if len(batch) > 7 else None
        dataset_nums = batch[8] if len(batch) > 8 else None

        # Prepare token embeddings for the cell sentence
        bs_emb = pl_module.pe_embedding(bs_indices)
        bs_emb = torch.nn.functional.normalize(bs_emb, dim=2)
        bs_emb[:, 0, :] = pl_module.cls_token.expand(bs_emb.size(0), -1)

        # Forward to get CLS embedding and optional dataset embedding
        gene_output, embedding, dataset_emb = pl_module.forward(
            bs_emb,
            mask=masks.bool(),
            counts=batch_sentences_counts if hasattr(pl_module.cfg.model, "counts") and pl_module.cfg.model.counts else None,
            dataset_nums=dataset_nums,
            token_indices=bs_indices,
        )

        # Get embeddings for task gene indices
        X_emb_initial = pl_module.pe_embedding(X_indices)
        X_emb = pl_module.gene_embedding_layer(X_emb_initial)

        # Build combine tensor similar to shared_step
        z = embedding.unsqueeze(1).repeat(1, X_emb.shape[1], 1)
        if getattr(pl_module, "z_dim_rd", 0) == 1:
            # Compute mu from Y ignoring zeros
            Y = Y_targets
            mu = torch.nan_to_num(
                torch.nanmean(Y.float().masked_fill(Y == 0, float("nan")), dim=1),
                nan=0.0,
            )
            reshaped_counts = mu.unsqueeze(1).unsqueeze(2).repeat(1, X_emb.shape[1], 1)
            combine = torch.cat((X_emb, z, reshaped_counts), dim=2)
        else:
            combine = torch.cat((X_emb, z), dim=2)

        if getattr(pl_module, "dataset_token", None) is not None and dataset_emb is not None:
            ds_emb = pl_module.dataset_embedder(dataset_emb)
            ds_emb = ds_emb.unsqueeze(1).repeat(1, X_emb.shape[1], 1)
            combine = torch.cat((combine, ds_emb), dim=2)

        decs = pl_module.binary_decoder(combine).squeeze(-1)  # [B, task_num]

        # Build per-batch gene union matrices
        X_cpu = X_indices.detach().cpu().numpy()
        B, T = X_cpu.shape
        # gene union for this batch
        gene_union = np.unique(X_cpu.reshape(-1))
        gene_to_col = {int(g): i for i, g in enumerate(gene_union.tolist())}
        pred_matrix = np.zeros((B, len(gene_union)), dtype=np.float32)
        true_matrix = np.zeros((B, len(gene_union)), dtype=np.float32)
        decs_cpu = decs.detach().cpu().numpy()
        Y_cpu = Y_targets.detach().cpu().numpy()
        for i in range(B):
            for j in range(T):
                g = int(X_cpu[i, j])
                c = gene_to_col[g]
                pred_matrix[i, c] = decs_cpu[i, j]
                true_matrix[i, c] = Y_cpu[i, j]
        return pred_matrix, true_matrix, idxs, gene_union

    def _run_probes_once(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # Import here to avoid circulars
        try:
            from ..utils import compute_pearson_delta, compute_perturbation_ranking_score
        except Exception as e:
            logger.debug(f"ProbeEvalCallback: utilities unavailable: {e}")
            return
        if not self._ensure_dev():
            return

        # Prepare inference dataloader
        try:
            from ..data.loader import create_dataloader
        except Exception as e:
            logger.debug(f"ProbeEvalCallback: dataloader unavailable: {e}")
            return

        # Optionally subsample to keep probe light
        adata = self._dev_adata
        pert_col = getattr(self._pert_conf, "pert_col", "gene") if self._pert_conf else "gene"
        ctrl_label = getattr(self._pert_conf, "ctrl_label", "non-targeting") if self._pert_conf else "non-targeting"
        # Shallow copy subset if very large
        try:
            max_cells = int(getattr(getattr(self.cfg, "validations"), "max_cells", 4096))
        except Exception:
            max_cells = 4096
        if adata.n_obs > max_cells:
            adata_sample = adata[:max_cells].copy()
        else:
            adata_sample = adata

        dl = create_dataloader(self.cfg, workers=2, adata=adata_sample, adata_name=self._dev_name, shuffle=False, precision=None)

        # Iterate a few batches and aggregate
        max_batches = 8
        num_batches = 0
        pd_vals = []
        pr_vals = []
        for batch in dl:
            pred, true, idxs, gene_union = self._compute_batch_preds(pl_module, batch)
            labels = adata_sample.obs[pert_col].values[idxs]

            # controls mask
            ctrl_mask = labels == ctrl_label
            if ctrl_mask.sum() == 0 or ctrl_mask.sum() == len(labels):
                # skip degenerate batches
                continue
            pred_ctrl = pred[ctrl_mask]
            true_ctrl = true[ctrl_mask]

            # Compute pearson_delta on per-batch union gene set
            try:
                pd_val = compute_pearson_delta(pred, true, pred_ctrl, true_ctrl)
                if np.isfinite(pd_val):
                    pd_vals.append(float(pd_val))
            except Exception:
                pass

            # Build minimal AnnData for perturbation ranking score
            try:
                ad_real = ad.AnnData(X=true, dtype=np.float32)
                ad_real.obs[pert_col] = labels
                ad_pred = ad.AnnData(X=pred, dtype=np.float32)
                ad_pred.obs[pert_col] = labels
                pr_val = compute_perturbation_ranking_score(ad_pred, ad_real, pert_col=pert_col, ctrl_pert=ctrl_label)
                if np.isfinite(pr_val):
                    pr_vals.append(float(pr_val))
            except Exception:
                pass

            num_batches += 1
            if num_batches >= max_batches:
                break

        if len(pd_vals) == 0 and len(pr_vals) == 0:
            return
        metrics = {}
        if len(pd_vals) > 0:
            metrics["probes/pearson_delta"] = float(np.mean(pd_vals))
        if len(pr_vals) > 0:
            metrics["probes/pert_rank"] = float(np.mean(pr_vals))
        if trainer.logger is not None and len(metrics) > 0:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._enabled:
            return
        self._val_calls += 1
        # Guard cadence with either (or both) configs
        try:
            de_mult = int(getattr(self._de_conf, "eval_interval_multiple", 10)) if self._de_conf else None
        except Exception:
            de_mult = None
        try:
            pr_mult = int(getattr(self._pert_conf, "eval_interval_multiple", 10)) if self._pert_conf else None
        except Exception:
            pr_mult = None
        should_run = False
        if de_mult is not None and self._val_calls % max(1, de_mult) == 0:
            should_run = True
        if pr_mult is not None and self._val_calls % max(1, pr_mult) == 0:
            should_run = True
        if should_run:
            self._run_probes_once(trainer, pl_module)
