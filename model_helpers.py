import torch
from typing import List
import os

def save_model(
    filename: str,
    model: torch.nn.Module,
    data_parallel: bool=False
):
    s = filename.split("/")[:-1]
    f = os.path.join(*s)
    if not os.path.exists(f):
        os.makedirs(f)
    if not data_parallel:
        torch.save(model.state_dict(), filename)
    else:
        torch.save(model.module.state_dict(), filename)


def load_model(filename: str, model: torch.nn.Module) -> torch.nn.Module:
    model = model.load_state_dict(torch.load(filename))
    model = model.eval()
    return model
