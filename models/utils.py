import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
