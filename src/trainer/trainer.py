import torch
import numpy as np

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def make_opt_step(self, 
                      batch_idx, 
                      batch,
                      metrics,
                      loss_name, 
                      optimizer, 
                      lr_scheduler):
        self.grad_scaler.scale(batch[loss_name]).backward()
        if ((batch_idx + 1) % self.iters_to_accumulate == 0) or (
            (batch_idx + 1) == self.epoch_len
        ):
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
            optimizer.zero_grad()

            metrics.update(f"grad_norm_{model_name}", self._get_grad_norm(modules))

            if lr_scheduler is not None:
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
                batch_idx,
                batch,
                metrics,
                "disc_loss",
                self.disc_optimizer,
                self.disc_lr_scheduler
            )

        with torch.autocast(
            self.device, dtype=mixed_precision, enabled=mixed_precision is not torch.float32
        ):
            batch.update(self.model.forward_generator(**batch))
            all_losses = self.gen_criterion(**batch)
            batch.update(all_losses)
            if self.is_train:
                batch["gen_loss"] /= self.iters_to_accumulate
        if self.is_train:
            self.make_opt_step(
                batch_idx,
                batch,
                metrics,
                "gen_loss",
                self.gen_optimizer,
                self.gen_lr_scheduler
            )

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            saved_loss = batch[loss_name].item()
            if self.is_train:
                saved_loss *= self.iters_to_accumulate
            metrics.update(loss_name, saved_loss)

        with torch.no_grad():
            for met in metric_funcs:
                metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, sample_rate, num_samples, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(num_samples=num_samples, **batch)
            self.log_audio(sample_rate=sample_rate, num_samples=num_samples, **batch)
        else:
            # Log Stuff
            self.log_spectrogram(num_samples=num_samples, **batch)
            self.log_audio(sample_rate=sample_rate, num_samples=num_samples, **batch)

    def log_spectrogram(self, num_samples, **batch):
        real_spectrogram = batch["spectrogram"][0:num_samples].squeeze(1).detach().cpu()
        generated_spectrogram = batch["generated_spectrogram"][0:num_samples].squeeze(1).detach().cpu()

        for i, (real_sample, generated_sample) in enumerate(zip(real_spectrogram, generated_spectrogram)):
            self.writer.add_image(f"real spectrogram {i + 1}", plot_spectrogram(real_sample))
            self.writer.add_image(f"generated spectrogram {i + 1}", plot_spectrogram(generated_sample))

    def log_audio(self, sample_rate, num_samples, **batch):
        real_audio = batch["audio"][0:num_samples].squeeze(1).deatch().cpu().numpy()
        generated_audio = batch["generated"][0:num_samples].squeeze(1).detach().cpu().numpy()
        if np.max(np.abs(generated_audio), dim=-1) > 1:
            generated_audio /= np.max(np.abs(generated_audio), dim=-1, keepdims=True)

        for i, (real_sample, generated_sample) in enumerate(zip(real_audio, generated_audio)):
            self.writer.add_audio(f"real audio {i + 1}", real_sample, sample_rate=sample_rate)
            self.writer.add_audio(f"generated audio {i + 1}", generated_sample, sample_rate=sample_rate)

    def log_predictions(
        self,
        **batch,
    ):
        # TBD
        pass
