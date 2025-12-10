import os
import zipfile
from pathlib import Path

import torchaudio
import yadisk
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json


class VCTKDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        index_dir=None,
        dataset_url=None,
        speaker_id=None,
        *args,
        **kwargs,
    ):
        self.speaker_id = speaker_id
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        if index_dir is None:
            index_dir = self.data_root
        else:
            index_dir = Path(index_dir)
        index_path = index_dir / f"vctk_index_{speaker_id if speaker_id else 'all'}.json"

        if index_path.exists():
            index = read_json(index_path)
        else:
            os.makedirs(str(index_path.parent), exist_ok=True)
            index = self._create_index(index_path, dataset_url)

        super().__init__(index, *args, **kwargs)

    def _create_index(
        self,
        index_path: Path,
        dataset_url: None | str,
    ):
        index = []

        top_level_dir = ""
        if dataset_url is not None:
            if dataset_url.startswith("http"):
                y = yadisk.YaDisk()
                meta = y.get_public_meta(dataset_url)
                total_size = meta.size
                file_name = meta.name

                print(f"Downloading {file_name} ({total_size / 1e6:.2f} MB)")

                archive_path = self.data_root / file_name
                y.download_public(dataset_url, str(archive_path))

                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(self.data_root)
                    top_level_dir = set(
                        name.split("/")[0] for name in zip_ref.namelist() if name.strip()
                    )
                    assert (
                        len(top_level_dir) == 1 or top_level_dir == "transcriptions"
                    ), "Wrong format for inference dir"
                    if top_level_dir != "transcriptions":
                        top_level_dir = top_level_dir.pop()
                    else:
                        top_level_dir = ""

                os.remove(archive_path)
            else:
                raise RuntimeError("dataset path must be either URL or None")

        top_level_dir = "VCTK-Corpus" if dataset_url is None else top_level_dir
        voices_path = self.data_root / top_level_dir / "wav48"
        if self.speaker_id is not None:
            voices_path = voices_path / self.speaker_id
        text_path = self.data_root / top_level_dir / "txt"

        for item in tqdm(voices_path.rglob("*.wav")):
            speaker_id = str(item.parent).split("/")[-1]
            audio_tensor, sample_rate = torchaudio.load(str(item))
            cur_text_path = text_path / speaker_id / (item.stem + ".txt")
            with open(cur_text_path, "r") as txt_f:
                cur_text = txt_f.read().strip()
            data_instance = {
                "audio_path": str(item),
                "length": audio_tensor.size(-1) / sample_rate,
                "speaker_id": speaker_id,
                "text": cur_text,
            }
            index.append(data_instance)

        write_json(index, str(index_path))

        return index
