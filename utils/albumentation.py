from albumentations import Compose
from albumentations.pytorch import ToTensor
import numpy as np


def getAlbumTransformation(transformList) :
    transformList.append(ToTensor())
    data_transforms = Compose(transformList)
    return lambda x: data_transforms(image=np.array(x))['image']

