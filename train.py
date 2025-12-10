import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from speechbrain.inference.TTS import FastSpeech2

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    torch.cuda.empty_cache()

    project_config = OmegaConf.to_container(config, resolve=True)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger=logger, project_config=project_config, _recursive_=False)

    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    if config.get("acoustic_config", None) is not None:
        acoustic_model = FastSpeech2.from_hparams(
            source=config.acoustic_config.model_name,
            savedir=config.acoustic_config.save_dir,
            run_opts={"device": "cpu"}
        )
    else:
        acoustic_model = None

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, acoustic_model)

    # build model architecture, then print to console
    melspec_transformer = instantiate(config.melspec_transformer)
    model = instantiate(config.model, melspec_transformer=melspec_transformer).to(device)
    logger.info(model)
    if config.trainer.get("compile", False):
        assert not config.trainer.get("ts_compile", False)
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

    # get function handles of loss and metrics
    metrics = instantiate(config.metrics)

    trainer = Trainer(
        model=model,
        metrics=metrics,
        config=config,
        project_config=project_config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        acoustic_model=acoustic_model
    )

    trainer.train()


if __name__ == "__main__":
    main()
