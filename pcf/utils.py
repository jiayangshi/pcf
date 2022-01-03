'''
The function used to count the trainable parameters of a network
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)