import numpy as np
import pandas as pd
import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def make_opt_step(self, batch_idx, batch, metrics, loss_name, optimizer, lr_scheduler):
        self.grad_scaler.scale(batch[loss_name]).backward()
        if ((batch_idx + 1) % self.iters_to_accumulate == 0) or ((batch_idx + 1) == self.epoch_len):
            model_name = loss_name.split("_")[0]
            if model_name == "gen":
                modules = self.model.generator
            elif model_name == "disc":
                modules = [self.model.mpd, self.model.msd]
            else:
                raise NameError()

            self.grad_scaler.unscale_(optimizer)
            self._clip_grad_norm(modules)
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()

            metrics.update(f"grad_norm_{model_name}", self._get_grad_norm(modules))
            optimizer.zero_grad()

            if self.is_train and model_name == "disc":
                self.cur_disc_step += 1

            if lr_scheduler is not None:
                if model_name == "disc" and self.cur_disc_step < self.disc_to_gen_update_ratio:
                    return
                if self.scheduler_config is None or not self.scheduler_config.update_after_epoch:
                    lr_scheduler.step()

    def process_batch(self, batch, batch_idx: int, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        if self.is_train:
            mixed_precision = self.mixed_precision
        else:
            mixed_precision = torch.float32

        with torch.autocast(
            self.device, dtype=mixed_precision, enabled=mixed_precision is not torch.float32
        ):
            batch.update(self.model.forward_discriminator(**batch))
            all_losses = self.disc_criterion(**batch)
            batch.update(all_losses)
            if self.is_train:
                batch["disc_loss"] /= self.iters_to_accumulate
        if self.is_train:
            self.make_opt_step(
                batch_idx, batch, metrics, "disc_loss", self.disc_optimizer, self.disc_lr_scheduler
            )

        with torch.set_grad_enabled(self.cur_disc_step == self.disc_to_gen_update_ratio):
            with torch.autocast(
                self.device, dtype=mixed_precision, enabled=mixed_precision is not torch.float32
            ):
                batch.update(self.model.forward_generator(**batch))
                all_losses = self.gen_criterion(**batch)
                batch.update(all_losses)
                if self.is_train:
                    batch["gen_loss"] /= self.iters_to_accumulate
            if self.is_train and self.cur_disc_step == self.disc_to_gen_update_ratio:
                self.cur_disc_step = 0
                self.make_opt_step(
                    batch_idx, batch, metrics, "gen_loss", self.gen_optimizer, self.gen_lr_scheduler
                )

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            saved_loss = batch[loss_name].item()
            if self.is_train:
                saved_loss *= self.iters_to_accumulate
            metrics.update(loss_name, saved_loss)

        peak_val, _ = torch.max(torch.abs(batch["generated"]), dim=-1)
        exceeds_peak = peak_val > 1
        if torch.any(exceeds_peak) > 1:
            norm_factor = peak_val[exceeds_peak][..., None]
            batch["generated"][exceeds_peak] = batch["generated"][exceeds_peak] / norm_factor

        batch["metrics"] = {}
        with torch.no_grad():
            for met in metric_funcs:
                calculated_metric = met(**batch)
                if not isinstance(calculated_metric, list):
                    calculated_metric = [calculated_metric] * batch["generated"].size(0)
                batch["metrics"][met.name] = calculated_metric
                metrics.update(met.name, sum(calculated_metric) / len(calculated_metric))
        return batch
