


def norm(param,p=None):
    if not p: 
        param[1].grad.abs().mean() if param[1].grad != None else 0.
    return param[1].grad.norm(p=p) if param[1].grad != None else 0.

def model_param_filter(model_params, thresh=0., inclusive=False):
    """
    filter to only params with 'norm' below threshold
    if thresh is negative (e.g. -0.1), threshold is bottom -thresh (e.g. 10) percentile value of the norms
    """
    lis = []
    if thresh == -1.:
        # freeze every param
        return model_param_filter(model_params, float("inf"))
    if thresh < 0.:
        norms = [norm(y) for y in model_params]
        assert int(-thresh * len(norms)) > 0, "-thresh too small!"
        new_thresh = sorted(norms)[:int(-thresh * len(norms))][-1]
        return model_param_filter(model_params, new_thresh, True)
        
    for y in model_params:
        freeze = norm(y) <= thresh if inclusive else norm(y) < thresh
        if freeze:
            lis.append((norm(y), y))
    
    return lis

