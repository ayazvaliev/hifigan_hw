import os
import zipfile
from pathlib import Path

import yadisk
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset

class CustomDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        dataset_url=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_root (str): path to dataset dir
            name (str): partition name
            index_dir (str): path to index dir (convenient for kaggle as their dataset section is ronly)
            dataset_url (str): URL to dataset.
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        index = self._create_index(dataset_url=dataset_url)

        super().__init__(index, *args, **kwargs)

    def _create_index(
        self, dataset_url: None | str,
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
                        len(top_level_dir) == 1
                        or top_level_dir == "transcriptions"
                    ), "Wrong format for inference dir"
                    if top_level_dir != "transcriptions":
                        top_level_dir = top_level_dir.pop()
                    else:
                        top_level_dir = ""

                os.remove(archive_path)
            else:
                raise RuntimeError("dataset path must be either URL or None")

        transcriptions_path = self.data_root / top_level_dir / "transcriptions"

        for item in tqdm(transcriptions_path.iterdir()):
            with open(item, 'r') as txt_f:
                transcription = txt_f.read().rstrip()
            data_instance = {
                "text": transcription,
                "text_id": item.stem,
            }
            index.append(data_instance)

        return index
