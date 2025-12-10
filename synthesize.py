import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from speechbrain.inference.TTS import FastSpeech2


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    project_config = OmegaConf.to_container(config, resolve=True)

    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    if config.get("writer", None) is not None:
        writer = instantiate(config.writer, project_config)
        melspec_transformer = instantiate(config.melspec_transformer, _recursive_=False).to(device)
    else:
        writer = None
        melspec_transformer = None

    if config.get("acoustic_config", None) is not None:
        acoustic_model = FastSpeech2.from_hparams(
            source=config.acoustic_config.model_name,
            savedir=config.acoustic_config.save_dir,
            run_opts={"device": "cpu"}
        )
    else:
        acoustic_model = None

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)

    if config.prompt is not None:
        checkpoint = torch.load(config.inferencer.save_path, map_location=device, weights_only=False)
        if checkpoint.get("state_dict") is not None:
            sd = checkpoint["state_dict"]
        else:
            sd = checkpoint
        model.load_state_dict(sd)

        melspec_output, _, _, _ = acoustic_model.encode_text([config.prompt])
        generated = model(melspec_output).squeeze(0)
        torchaudio.save(str(Path(config.inferencer.save_path) / "generated.wav"), generated, sample_rate=config.inferencer.sr, format="wav")
        return

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, acoustic_model)

    if config.inferencer.get("compile", False):
        assert not config.trainer.get("ts_compile", False)
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

    # get metrics
    metrics = instantiate(config.metrics)

    # save_path for model predictions
    save_path = config.inferencer.get("save_path", None)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
        writer=writer,
        melspec_transformer=melspec_transformer
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        if logs[part] is None:
            print(f"{part}: No metrics were calculated")
            continue
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
