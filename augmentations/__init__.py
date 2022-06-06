from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single


def get_aug(name='simsiam', image_size=224, train=True, no_train_augs=False, train_classifier=None):
    if train==True:
        augmentation = SimSiamTransform(image_size, no_train_augs=no_train_augs)
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








