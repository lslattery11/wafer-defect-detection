import torch
import torchvision.transforms.functional as TF
import random

def wafer_train_transforms(
    p_threshold: float ,
    ):
    """
    data augmentation for training data.
    """
    def random_rotate(x:torch.Tensor):
        if random.random() > p_threshold:
            angle = random.randint(0,180)
            x = TF.rotate(x, angle)
        return x

    return random_rotate