from pathlib import Path
import torch


CKPT_DIR = "ckpts/"
for ckpt in Path(CKPT_DIR).iterdir():
    if str(ckpt).split(".")[-1] == "pth":
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        gen_sd = {".".join(k.split('.')[1:]) : val for k, val in checkpoint['state_dict'].items() if 'generator' in k}
        torch.save(gen_sd, Path(ckpt.parent) / (ckpt.stem + "_gen.pth"))