import torch

def get_model_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_params(model, params):
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)