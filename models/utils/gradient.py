


def norm(param,p=2):
    return param[1].grad.norm(p=p) if param[1].grad != None else 0.

def model_param_filter(model, thresh=0.):
    """
    filter to only params with 'norm' below threshold
    if thresh is negative (e.g. -0.1), threshold is bottom -thresh (e.g. 10) percentile value of the norms
    """
    lis = []
    if thresh < 0.:
        norms = [norm(y) for y in list(model.named_parameters())]
        assert int(-thresh * len(norms)) > 0, "-thresh too small!"
        new_thresh = sorted(norms)[:int(-thresh * len(norms))][-1]
        return model_param_filter(model, new_thresh)
        
    for y in list(model.named_parameters()): 
        if norm(y) <= thresh:
            lis.append((norm(y), y))
    
    return [x[1] for x in lis]

