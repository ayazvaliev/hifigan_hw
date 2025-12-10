import os
import zipfile
from pathlib import Path

import torchaudio
import yadisk
import pandas as pd
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json
from src.datasets.data_utils import normalize_text


class LJSpeechDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        name="train",
        train_test_split_ratio=None,
        index_dir=None,
        dataset_url=None,
        inference_mode=False,
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

        self.inference_mode = inference_mode

        if not inference_mode:
            if index_dir is None:
                index_dir = self.data_root
            else:
                index_dir = Path(index_dir)
            index_path = index_dir / "index.json"

            # each nested dataset class must have an index field that
            # contains list of dicts. Each dict contains information about
            # the object, including label, path, etc.
            if index_path.exists():
                index = read_json(index_path)
            else:
                os.makedirs(str(index_path.parent), exist_ok=True)
                index = self._create_index(name, index_path, dataset_url)
            if train_test_split_ratio is not None:
                assert (0 <= train_test_split_ratio) and (train_test_split_ratio <= 1)
                if name == "train":
                    index = index[int(len(index) * train_test_split_ratio):]
                else:
                    index = index[:int(len(index) * train_test_split_ratio)]
        else:
            index = self._create_index(
                index_path=None, dataset_url=dataset_url, write_to_disk=False
            )
        super().__init__(index, *args, **kwargs)

    def _create_index(
        self, index_path: Path, dataset_url: None | str, write_to_disk=True
    ):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
            path (str): path to yandex disk file
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []

        top_level_dir = ""
        if dataset_url is not None:
            print(f'Downloading from {dataset_url}')
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
                        or top_level_dir == "wavs"
                    ), "Wrong format for dir"
                    if top_level_dir != "wavs":
                        top_level_dir = top_level_dir.pop()
                    else:
                        top_level_dir = ""

                os.remove(archive_path)
            else:
                raise RuntimeError("dataset path must be either URL or None")

        if not self.inference_mode:
            top_level_dir = "LJSpeech-1.1" if dataset_url is None else top_level_dir
            metadata_df = pd.read_csv(self.data_root / top_level_dir / "metadata.csv", sep="|", header=None, index_col=None)
        else:
            metadata_df = None

        audio_path = self.data_root / top_level_dir / "wavs"

        for item in tqdm(audio_path.iterdir()):
            item_name = item.stem
            audio_tensor, sample_rate = torchaudio.load(str(item))
            num_frames = audio_tensor.size(-1)
            data_instance = {
                "audio_path": str(item),
                "length": num_frames / sample_rate,
            }
            if metadata_df is not None:
                data_instance["text"] = normalize_text(str(metadata_df.loc[metadata_df[0] == item_name, 2].iloc[0]))
            index.append(data_instance)

        # write index to disk
        if write_to_disk:
            write_json(index, str(index_path))

        return index
