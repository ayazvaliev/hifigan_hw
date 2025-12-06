import torch
from torch.nn.utils.rnn import pad_sequence

def collate_factory(config):
    def collate_fn(dataset_items: list[dict]):
        """
        Collate and pad fields in the dataset items.
        Converts individual items into a batch.

        Args:
            dataset_items (list[dict]): list of objects from
                dataset.__getitem__.
        Returns:
            result_batch (dict[Tensor]): dict, containing batch-version
                of the tensors.
        """
        # spectrogram [F, T] -> [T, F] -> [N, T, F] -> [N, F, T] NFT :O
        # audio [T] -> [N, T]
        spec_padding = config.melspec_transformer.config.pad_value
        text_padding = None # TODO
        batch = {}

        if "spectrogram" in dataset_items[0].keys():
            batch["spectrogram"] =  pad_sequence([elem["spectrogram"].transpose(0, 1) for elem in dataset_items], 
                                                  batch_first=True,
                                                  padding_value=spec_padding).transpose(1,2)
        if "audio" in dataset_items[0].keys():
            batch["audio"] = pad_sequence([elem["audio"] for elem in dataset_items],
                                          batch_first=True)
        if "text_encoded" in dataset_items[0].keys():
            batch["text_encoded"] =  pad_sequence([elem["text_encoded"] for elem in dataset_items],
                                                  batch_first=True,
                                                  padding_value=text_padding)

        excluded_keys = set(batch.keys())
        for k in dataset_items[0].keys():
            if k not in excluded_keys:
                batch[k] = [elem[k] for elem in dataset_items]
        
        return batch

    return collate_fn
