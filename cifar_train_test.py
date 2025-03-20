import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision.datasets import CIFAR10
from torchvision import transforms

batch_size=256
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - mean) / std
    return img

# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
    
test_transform = image_to_numpy
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
    image_to_numpy
])

cifar_dataset = CIFAR10('/tmp/cifar10/', download=True, transform=train_transform)
cifar_test = CIFAR10('/tmp/cifar10/', download=True, train=False, transform=test_transform)
cifar_val = CIFAR10('/tmp/cifar10/', download=True, transform=test_transform)

train_set, _ = torch.utils.data.random_split(cifar_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
_, val_set = torch.utils.data.random_split(cifar_val, [45000, 5000], generator=torch.Generator().manual_seed(42))

    
train_loader = data.DataLoader(train_set,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True,
                               collate_fn=numpy_collate,
                               num_workers=0,
                               )
val_loader   = data.DataLoader(val_set,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=0,
                               )
test_loader  = data.DataLoader(cifar_test,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=0,
                               )
