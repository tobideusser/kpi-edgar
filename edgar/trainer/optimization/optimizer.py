from importlib import import_module

import torch

OPTIMIZERS = {"adam": "torch.optim.Adam", "adamW": "transformers.AdamW", "adagrad": "torch.optim.Adagrad"}


class Optimizer:
    @classmethod
    def from_config(cls, type_: str, *args, **kwargs) -> torch.optim.Optimizer:
        try:
            callable_path = OPTIMIZERS[type_]
            parts = callable_path.split(".")
            module_name = ".".join(parts[:-1])
            class_name = parts[-1]
        except KeyError:
            raise KeyError(f'Optimizer "{type_}" is not implemented.')

        module = import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(*args, **kwargs)
