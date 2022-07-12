
import torchvision.transforms as T

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

default_normalization = T.Normalize(
    _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
    _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
)

class ImageBaseTransform():
    def __init__(self):
        # Assume input 224, 224        
        self.not_aug_transform = T.Compose([T.ToTensor()])

        self.transform = T.Compose([
            T.ToTensor(),
            default_normalization
        ])
        
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        not_aug_x = self.not_aug_transform(x)
        return x1, x2, not_aug_x


class ImageBaseTransformSingle():
    def __init__(self):
        self.transform = T.Compose([
            # T.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
            # T.CenterCrop(image_size),
            T.ToTensor(),
            default_normalization
        ])

    def __call__(self, x):
        return self.transform(x)