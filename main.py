import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms

from augmentations import AugmixDataset
from WideResNet_pytorch.wideresnet import WideResNet

PATH = "./ckpt/augmix.ckpt"

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def test(model, test_data, batch_size):
    test_loader = torch.utils.data.DataLoader(
                   test_data,
                   batch_size=batch_size,
                   shuffle=False,
                   pin_memory=True)
    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0
        for i, (images, targets) in enumerate(test_loader):
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            logits = torch.argmax(logits, dim=-1)
            correct += targets.eq(logits).sum().item()
            total += targets.size()[0]
    acc = correct * 1. / total
    return acc

def main():
    torch.manual_seed(2022)
    np.random.seed(2022)
    epochs = 100
    js_loss = True
    model_load = False
    lmbda = 12
    batch_size = 256
    dataset = "CIFAR-10"
    os.makedirs('./ckpt/', exist_ok=True)

    # 1. dataload
    # basic augmentation & preprocessing
    train_base_aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
    ]
    preprocess = [
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
    ]
    if js_loss: 
        train_transform = transforms.Compose(train_base_aug)
    else:
        train_transform = transforms.Compose(train_base_aug + preprocess)
    test_transform = transforms.Compose(preprocess)
    # load data
    if dataset == "CIFAR-10":
        train_data = datasets.CIFAR10('./data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10('./data/cifar', train=False, transform=test_transform, download=True)
    else:
        train_data = datasets.CIFAR100('./data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100('./data/cifar', train=False, transform=test_transform, download=True)
    if js_loss:
        train_data = AugmixDataset(train_data, test_transform)
    train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True)
    # 2. model
    # wideresnet 40-2
    n_classes = 10 if dataset=="CIFAR-10" else 100
    model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, drop_rate=0.0)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.SGD(
                  model.parameters(),
                  0.1,
                  momentum=0.9,
                  weight_decay=0.0005,
                  nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader), eta_min=1e-6, last_epoch=-1)

    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    if not model_load:
        # training model with cifar
        model.train()
        losses = []
        start = time.time()
        for epoch in range(epochs):
            for i, (images, targets) in tqdm(enumerate(train_loader)):
                if js_loss:
                    images = torch.cat(images, axis=0)
                images, targets = images.cuda(), targets.cuda()
                optimizer.zero_grad()
                if js_loss:
                    raise NotImplementedError
                else:
                    logits = model(images)
                    loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            if epoch == 0:
                print("Time takes for 1 epochs: %s" %(time.time()-start))
            print("Train Loss: {:.4f}".format(loss.item()))
            torch.save({
                "epoch": epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses
            }, PATH)
    else:
        ckpt = torch.load(PATH)
        model.load_state_dict(ckpt["model_state_dict"])
    # calculate clean error
    acc = test(model, test_data, 2000)
    print("[TEST] CLEAN Accuracy : {:.2f}".format(acc))
    
    # evaluate on cifar-c
    CEs = []
    for corruption in CORRUPTIONS:
        test_data.data = np.load('./data/cifar/'+dataset+'-C/%s.npy' % corruption)
        test_data.targets = torch.LongTensor(np.load('./data/cifar/'+dataset+'-C/labels.npy'))
        acc = test(model, test_data, 2000)
        print("%s: %f" %(corruption, 1-acc))
        CEs.append(1-acc)
    mCE = sum(CEs) / len(CEs)
    print("[TEST] mCE : {:.2f}".format(mCE))

if __name__=="__main__":
    main()
