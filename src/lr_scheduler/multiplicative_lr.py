import torch.optim as optim


class MultiplicativeLR:
    def __init__(self, optimizer, coeff):
        self._lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            optimizer=optimizer, lr_lambda=lambda step: coeff
        )

    def __getattr__(self, name):
        return getattr(self._lr_scheduler, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        if hasattr(self, "_lr_scheduler"):
            setattr(self._lr_scheduler, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name.startswith("_"):
            super().__delattr__(name)
            return

        if hasattr(self, "_lr_scheduler"):
            delattr(self._lr_scheduler, name)
        else:
            super().__delattr__(name)
