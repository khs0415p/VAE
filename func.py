import torch

def save_model(path, epoch, loss, model_state, optimizer_state, scheduler_state=None):
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state
    }, path)


