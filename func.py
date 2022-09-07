import torch

def save_model(path, epoch, loss, model_state, optimizer_state, scheduler_state=None):
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state
    }, path)


def make_z(mean, log_var):
    std = torch.exp(0.5 * log_var)
    z = torch.randn_like(mean)*std + mean
    return z.numpy()