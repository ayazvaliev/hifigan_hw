from pathlib import Path

import torch
import torchaudio
from torch_audiomentations import PeakNormalization
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        melspec_transformer=None,
        writer=None,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model_ = model
        self.batch_transforms = batch_transforms
        self.torchscript = self.cfg_trainer.get("ts_compile", False)

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path
        self.melspec_transformer = melspec_transformer
        self.writer = writer

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=writer,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

        if self.torchscript:
            self.model = torch.jit.script(self.model_)
        else:
            self.model = self.model_

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            if self.writer is not None:
                self.writer.set_step(0, mode=part)
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part_save_path):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update({"generated": outputs})

        peak_val, _ = torch.max(torch.abs(batch["generated"]), dim=-1)
        exceeds_peak = peak_val > 1
        if torch.any(exceeds_peak) > 1:
            norm_factor = peak_val[exceeds_peak][..., None]
            batch["generated"][exceeds_peak] = batch["generated"][exceeds_peak] / norm_factor

        batch["metrics"] = {}
        for met in self.metrics["inference"]:
            calculated_metric = met(**batch)
            if not isinstance(calculated_metric, list):
                calculated_metric = [calculated_metric] * batch["generated"].size(0)
            batch["metrics"][met.name] = calculated_metric

            metrics.update(met.name, sum(calculated_metric) / len(calculated_metric))

        if part_save_path is not None:
            batch["generated"] = batch["generated"].cpu()
            for wav, text_id in zip(batch["generated"], batch["text_id"]):
                save_name = part_save_path / f"{text_id}.wav"
                torchaudio.save(save_name, wav, sample_rate=self.config.inferencer.sr, format="wav")

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            part_save_path = self.save_path / part
            part_save_path.mkdir(exist_ok=True, parents=True)
        else:
            part_save_path = None

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    metrics=self.evaluation_metrics,
                    part_save_path=part_save_path,
                )
            if self.writer is not None:
                batch["generated_spectrogram"] = self.melspec_transformer(
                    batch["generated"].squeeze(1)
                )
                self._log_batch(batch_idx, batch, sample_rate=self.config.inferencer.sr)
                self._log_scalars(self.evaluation_metrics)

        ret_none = self.evaluation_metrics is None or self.evaluation_metrics.empty
        return self.evaluation_metrics.result() if not ret_none else None
