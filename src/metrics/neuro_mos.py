import os
import urllib

import torch
import torchaudio

from src.metrics.base_metric import BaseMetric
from src.model import Wav2Vec2MOS

PATH = os.path.join(os.path.expanduser("~"), ".cache/wv_mos/wv_mos.ckpt")


class WMOS(BaseMetric):
    def __init__(self, device, name=None, **kwargs):
        self.cuda_flag = device == "cuda"
        if not os.path.exists(PATH):
            os.makedirs(os.path.dirname(PATH), exist_ok=True)
            urllib.request.urlretrieve(
                "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1", PATH
            )
        self.model = Wav2Vec2MOS(PATH, cuda=self.cuda_flag)
        super().__init__(name=name)

    def __call__(self, generated: torch.Tensor, **batch):
        generated = torchaudio.functional.resample(generated.detach().cpu(), 22050, 16000).numpy()
        mos_vals = []
        for generated_sample in generated:
            x = self.model.processor(
                generated_sample, return_tensors="pt", padding=True, sampling_rate=16000
            ).input_values
            with torch.no_grad():
                if self.cuda_flag:
                    x = x.cuda()
            res = self.model.forward(x).mean().item()
            mos_vals.append(res)
        return mos_vals
