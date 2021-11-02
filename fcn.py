import numpy as np
import torch
import torch.nn as nn


import time
import os

###################
# hyperparameters #
###################
EPOCH_NUM = 1000
BATCH_SIZE = 32
CLASS_NUM = 22
SAVE_IMAGE_INTERVAL = 500
SAVE_MODEL_INTERVAL = 500
LEARING_RATE = 1e-4

VAL_INTERVAL = 5

MODEL_TYPE = '8'
LOAD_MODEL = False
LOAD_MODEL_PATH = "/home/jin/workspace/AI502/FCN/model/20211101-225918/20211101-230357_32_41.pt"

DATASET_DIR = f'{os.path.dirname(os.path.abspath(__file__))}/dataset/VOC'
LOGPATH = f'{os.path.dirname(os.path.abspath(__file__))}/log/{time.strftime("%Y%m%d-%H%M%S")}_{MODEL_TYPE}'
MODELPATH = f'{os.path.dirname(os.path.abspath(__file__))}/model/{time.strftime("%Y%m%d-%H%M%S")}'
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(LOGPATH, exist_ok=True)
os.makedirs(MODELPATH, exist_ok=True)

##################
# 0. make logger #
##################
from torch.utils.tensorboard import SummaryWriter
import torchvision

class Logger():
    """
    using tensorwach logger to log training progress
    """
    def __init__(self, log_dir, run_browser=False):
        self.writer = SummaryWriter(log_dir)

        if run_browser:
            from tensorboard import program
            import webbrowser as wb
            # learning visualizer
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', log_dir])
            url = tb.launch()
            wb.open_new(url)

    def add_image(self, name, images):
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(name, img_grid) # 3, H, W

    def add_graph(self, network, input):
        self.writer.add_graph(network, input)

    def add_scalar(self, name, values, steps):
        self.writer.add_scalar(name, values, steps)

    def add_histogram(self, name, values, steps):
        self.writer.add_histogram(name, values, steps)

    def print_progress(self, progress, total_step, loss):
        print(time.strftime("%Y%m%d-%H%M%S") + f"====progress: {progress}/{total_step}, loss: {loss}")

    def close(self):
        self.writer.close()

######################
# 1. make dataloader #
######################
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random

class VOCSegDataLoader(VOCSegmentation):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
        'contour'
    ])

    def __init__(self, root, year, image_set, download, transform=None, target_transform=None, transforms=None, crop_size=(224,224)):
        super(VOCSegDataLoader, self).__init__(root=root, 
                                                year=year, 
                                                image_set=image_set, 
                                                download=download, 
                                                transform = transform, 
                                                target_transform = target_transform, 
                                                transforms = transforms)

        self.crop_size = crop_size

    def __len__(self):
        return len(self.images)

    def custom_transform(self, image, mask):
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = np.array(mask)
        return image, mask

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        img = Image.open(self.images[index]).convert("RGB")
        lbl = Image.open(self.masks[index])
        
        # if resize isn't taken, batch training is impossible
        resize = transforms.Resize(size = self.crop_size, interpolation=Image.NEAREST)
        img = resize(img)
        lbl = resize(lbl)

        img, lbl = self.custom_transform(img, lbl)

        # contour(255) -> 21
        lbl[lbl==255] = 21

        # make one-hot encoding - 1 channel -> class_num channels
        h, w = lbl.shape
        target = np.zeros((len(self.class_names), h, w), dtype=np.float32)
        for c in range(len(self.class_names)):
            target[c][lbl == c] = 1

        target = torch.from_numpy(target).float()

        return img, target

train_data_loader = DataLoader(VOCSegDataLoader(root=DATASET_DIR, 
                                          year='2012', 
                                          image_set='train', 
                                          download=False, 
                                          transform=None, 
                                          target_transform=None,
                                          transforms=None),
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

val_data_loader = DataLoader(VOCSegDataLoader(root=DATASET_DIR,
                                          year='2012', 
                                          image_set='val', 
                                          download=False, 
                                          transform=None, 
                                          target_transform=None,
                                          transforms=None),
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

###########################
# 2. get pretrained model #
###########################
print("==========================================")
print(f"torch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print("==========================================")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import torchvision.models as models

model_vgg = models.vgg16(pretrained=True).features.to(device).eval()

feature_extractor = nn.Sequential(*list(model_vgg.children()))

# fix parameter
for param in feature_extractor.parameters():
    param.requires_grad = False

##########################
# 3. Implement FCN model #
##########################

class FCNBase(nn.Module):
    def __init__(self, feature_extractor, class_num = 22):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.class_num = class_num

    def forward(self, x):
        maxpool = []
        for name, module in self.feature_extractor._modules.items():
            '''
            Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): ReLU(inplace=True)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): ReLU(inplace=True)
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (8): ReLU(inplace=True)
            (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (11): ReLU(inplace=True)
            (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (13): ReLU(inplace=True)
            (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (15): ReLU(inplace=True)
            (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (18): ReLU(inplace=True)
            (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (20): ReLU(inplace=True)
            (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (22): ReLU(inplace=True)
            (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (25): ReLU(inplace=True)
            (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (27): ReLU(inplace=True)
            (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (29): ReLU(inplace=True)
            (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
            '''
            x = module(x)
            if name == '4' or name == '9' or name == '16' or name == '23' or name == '30':
                maxpool.append(x)
                
        return maxpool

    def init_weights(self, m):
        """
        function to initialize weights
        """
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

class FCNFriends(FCNBase):
    """
    FCN model that can use 8, 16, 32
    Args:
        FCNBase (nn.Module): base model: vgg16
    """
    def __init__(self,feature_extractor, class_num = 22, model_type = "32"):
        super().__init__(feature_extractor, class_num)

        self.model_type = model_type

        if not (self.model_type == "32" or self.model_type == "16" or self.model_type == "8"):
            raise ValueError("model_type must be 32, 16 or 8")

        self.convtranspose1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0)
        self.bn1            = nn.BatchNorm2d(512)
        self.convtranspose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.bn2            = nn.BatchNorm2d(256)
        self.convtranspose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.bn3            = nn.BatchNorm2d(128)
        self.convtranspose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.bn4            = nn.BatchNorm2d(64)
        self.convtranspose5 = nn.ConvTranspose2d(64, self.class_num, kernel_size=2, stride=2, padding=0)

        self.relu = nn.ReLU(inplace=True)

        self.output1 = nn.Sequential(
                                    self.convtranspose1,
                                    self.relu,
                                    self.bn1,
                                    )
        self.output2 = nn.Sequential(
                                    self.convtranspose2,
                                    self.relu,
                                    self.bn2,
                                )
        self.output3 = nn.Sequential(
                                    self.convtranspose3,
                                    self.relu,
                                    self.bn3,
                                    )   
        self.output4 = nn.Sequential(
                                    self.convtranspose4,
                                    self.relu,
                                    self.bn4,
                                    )
        self.output5 = nn.Sequential(
                                    self.convtranspose5
                                    )

        self.output1.apply(self.init_weights)
        self.output2.apply(self.init_weights)
        self.output3.apply(self.init_weights)
        self.output4.apply(self.init_weights)
        self.output5.apply(self.init_weights)

    def forward(self, x):
        layers = super().forward(x) # len 5 list of tensors, each of size (batch_size, 16*2^n, 224*2^-1, 224*2^-1)

        x = self.output1(layers[4]) # torch.Size([1, 512, 7, 7]) -> torch.Size([1, 512, 14, 14])

        if self.model_type == "8" or self.model_type == "16":
            x = x + layers[3]    # torch.Size([1, 512, 14, 14]) -> torch.Size([1, 512, 14, 14])

        x = self.output2(x) # torch.Size([1, 512, 14, 14]) -> torch.Size([1, 256, 28, 28])

        if self.model_type == "8":
            x = x + layers[2]   # torch.Size([1, 256, 28, 28]) -> torch.Size([1, 256, 28, 28])

        x = self.output3(x) # torch.Size([1, 128, 56, 56])
        x = self.output4(x) # torch.Size([1, 64, 112, 112])
        x = self.output5(x) # torch.Size([1, 22, 224, 224])

        return x

############
# 4. utils #
############

# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(CLASS_NUM):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))

    ious = torch.FloatTensor(ious)

    is_nan = torch.isnan(ious)
    ious[is_nan] = 0
    return ious.mean()

def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total   

def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

def generate_colors(num_classes):
    colors = []
    colors.append((0, 0, 0)) # __background__
    for i in range(1, num_classes):
        colors.append(generate_random_color())

    colors[-1] = (255, 255, 255) # __contour__

    return colors

COLOR_MAP = generate_colors(CLASS_NUM)

def mask_image(mask):

    mask_image = np.zeros((3, mask.shape[0], mask.shape[1])) # (3, h, w)

    for i in range(CLASS_NUM):
        mask_image[0, mask == i] = COLOR_MAP[i][0]
        mask_image[1, mask == i] = COLOR_MAP[i][1]
        mask_image[2, mask == i] = COLOR_MAP[i][2]

    return mask_image

##################
# 5. Train model #
##################
import torch.optim as optim

def train(epoch, model, optimizer, data_loader, loss_func, logger=None):
    model.train()

    for i, (img, target) in enumerate(data_loader):
        img = img.to(device)
        target = target.to(device)

        output = model(img)
        loss = loss_func(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.print_progress(progress=epoch*len(train_data_loader) + i, total_step=EPOCH_NUM*len(train_data_loader), loss=loss.item())

        logger.add_scalar('Loss', loss.item(), steps=epoch*len(train_data_loader) + i)
        ious = iou(output.argmax(axis=1), target.argmax(axis=1))
        logger.add_scalar('Mean IOU', ious.detach().cpu() , steps=epoch*len(train_data_loader) + i)
        logger.add_scalar('Pixel accuary', pixel_acc(output.argmax(axis=1), target.argmax(axis=1)).detach().cpu(), steps=epoch*len(train_data_loader) + i)

        if (epoch*len(train_data_loader) + i) % SAVE_IMAGE_INTERVAL == 0:
            logger.add_image(f'{epoch*len(train_data_loader) + i}_input', torch.from_numpy(mask_image(output[0].argmax(axis=0).detach().cpu().numpy())))
            logger.add_image(f'{epoch*len(train_data_loader) + i}_target', torch.from_numpy(mask_image(target[0].argmax(axis=0).detach().cpu().numpy())))


        if (epoch*len(train_data_loader) + i) % SAVE_MODEL_INTERVAL == 0:
                torch.save({
                    'network_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_num': epoch
                }, f"{MODELPATH}/{time.strftime('%Y%m%d-%H%M%S')}_{model.model_type}_{epoch}.pt")

def val(epoch, model, data_loader, loss_func, logger=None):
    model.eval()

    print("=========Start validation=========")

    total_loss = 0
    total_iou = 0
    total_pixel_acc = 0
    total_num = 0
    for i, (img, target) in enumerate(data_loader):
        img = img.to(device)
        target = target.to(device)

        output = model(img)

        loss = loss_func(output, target)
        total_loss += loss.item()
        total_iou += iou(output.argmax(axis=1), target.argmax(axis=1)).item()
        total_pixel_acc += pixel_acc(output.argmax(axis=1), target.argmax(axis=1)).item()
        total_num += 1

        logger.print_progress(progress=i, total_step=len(train_data_loader), loss=loss.item())

    logger.add_scalar('val_loss', total_loss/total_num, epoch)
    logger.add_scalar('val_iou', total_iou/total_num, epoch)
    logger.add_scalar('val_pixel_acc', total_pixel_acc/total_num, epoch)

    print("=========End validation=========")

fcn = FCNFriends(feature_extractor, class_num = 22, model_type=MODEL_TYPE).to(device)

optimizer = optim.SGD(fcn.parameters(), lr=LEARING_RATE, momentum=0.9)
criterion = nn.BCEWithLogitsLoss().to(device)
logger = Logger(LOGPATH, run_browser=False)
logger.add_graph(fcn, torch.rand(1, 3, 224, 224).to(device))

start_epoch = 0

# load parameter
if(LOAD_MODEL):
    print(f"{LOAD_MODEL_PATH} loading...")
    checkpoint = torch.load(LOAD_MODEL_PATH, map_location=device)
    fcn.load_state_dict(checkpoint['network_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch_num']
    print(f"{LOAD_MODEL_PATH} loading success!!!")

print("=========Start training=========")

for epoch in range(start_epoch, EPOCH_NUM):

    train(epoch=epoch, model=fcn, optimizer=optimizer, data_loader=train_data_loader, loss_func=criterion, logger=logger)

    if epoch % VAL_INTERVAL == 0:
        val(epoch=epoch, model=fcn, data_loader=val_data_loader, loss_func=criterion, logger=logger)

logger.close()