import torch


def cuda_to_cpu(model, path_model_best, save_path='tmps/model_best.pth.tar.cpu'):
    '''load model and convert model to cpu
    import torch
    model_dict = torch.load('model_best.pth.tar.cpu')
    model = model_dict['model']
    model.load_state_dict(model_dict['state_dict'])
    '''
    checkpoint = torch.load(path_model_best)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)
    model.cpu()

    model_dict = {}
    model_dict['model'] = model
    model_dict['state_dict'] = model.state_dict()

    torch.save(model_dict, save_path)

    return save_path