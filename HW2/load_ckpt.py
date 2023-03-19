import torch


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path)
    return ckpt


ckpt = load_checkpoint("HW1/Slot.pt")
ckpt2 = load_checkpoint("HW1/ckpt/intent/Intent.pt")
print(ckpt['acc'])
print(ckpt2['acc'])
