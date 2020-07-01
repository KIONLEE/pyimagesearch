root = './'
import sys
sys.path.append(root)

import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from data import VOCDetection
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Configurations
run_name = 'vgg16'          # experiment name.
ckpt_root = 'checkpoints'   # from/to which directory to load/save checkpoints.
data_root = 'dataset'       # where the data exists.
pretrained_backbone_path = 'weights/vgg_features.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001          # learning rate
batch_size = 64     # batch_size
last_epoch = 1      # the last training epoch. (defulat: 1)
max_epoch = 200     # maximum epoch for the training.

num_boxes = 2       # the number of boxes for each grid in Yolo v1.
num_classes = 20    # the number of classes in Pascal VOC Detection.
grid_size = 7       # 3x224x224 image is reduced to (5*num_boxes+num_classes)x7x7.
lambda_coord = 7    # weight for coordinate regression loss.
lambda_noobj = 0.5  # weight for no-objectness confidence loss.

ckpt_dir = os.path.join(root, ckpt_root)
makedirs(ckpt_dir)

train_dset = VOCDetection(root=data_root, split='train')
train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

test_dset = VOCDetection(root=data_root, split='test')
test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)


# Problem 1. Implement Architecture
class Yolo(nn.Module):
    def __init__(self, grid_size, num_boxes, num_classes):
        super(Yolo, self).__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.features = nn.Sequential(
            # implement backbone network here.
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.detector = nn.Sequential(
            # implement detection head here.
            nn.Linear(in_features=7*7*512, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.5, inplace=False),
            nn.Linear(in_features=4096, out_features=self.S*self.S*(self.B*5+self.C), bias=True)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.detector(x)
        x = F.sigmoid(x)
        x = x.view(-1, self.S, self.S, self.B*5+self.C)
        return x

model = Yolo(grid_size, num_boxes, num_classes)
model = model.to(device)
pretrained_weights = torch.load(pretrained_backbone_path)
model.load_state_dict(pretrained_weights)
# It should print out <All keys matched successfully> when you implemented VGG correctly.

# Freeze the backbone network.
model.features.requires_grad_(False)
model_params = [v for v in model.parameters() if v.requires_grad is True]
optimizer = optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=5e-4)

# Load the last checkpoint if exits.
ckpt_path = os.path.join(ckpt_dir, 'last.pth') 

if os.path.exists(ckpt_path): 
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    last_epoch = ckpt['epoch'] + 1
    print('Last checkpoint is loaded. start_epoch:', last_epoch)
else:
    print('No checkpoint is found.')


# Problem 2. Implement Architecture
class Loss(nn.Module):
    def __init__(self, grid_size=7, num_bboxes=2, num_classes=20):
        """ Loss module for Yolo v1.
        Use grid_size, num_bboxes, num_classes information if necessary.

        Args:
            grid_size: (int) size of input grid.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
        """
        super(Loss, self).__init__()
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Use this function if necessary.

        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss.

        Args:
            pred_tensor (Tensor): predictions, sized [batch_size, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor (Tensor):  targets, sized [batch_size, S, S, Bx5+C].
        Returns:
            loss_xy (Tensor): localization loss for center positions (x, y) of bboxes.
            loss_wh (Tensor): localization loss for width, height of bboxes.
            loss_obj (Tensor): objectness loss.
            loss_noobj (Tensor): no-objectness loss.
            loss_class (Tensor): classification loss.
        """
        # Write your code here
        def center_to_ltrb(_tensor):
            """ Transform tensor from 'center' to 'left-top, right-bottom'

            Args:
                _tensor (Tensor) : original, sized [filtered x B, 5], 5=len([x, y, w, h, conf]).
            Returns:
                tensor_ltrb (Tensor) : for computing iou, sized [filtered x B, 5] where we have 'filtered' cells vary in context, 5=len([x1, y1, x2, y2, conf]).
            """
            tensor_ltrb = torch.zeros_like(_tensor).to(device)
            # As in encoder function, both w,h are in image-size and both x,y are in cell-size
            cell_size = 1./self.S
            tensor_ltrb[:, :2] = _tensor[:, :2] * cell_size - _tensor[:, 2:4] * .5 # compute x1, y1
            tensor_ltrb[:, 2:4] = _tensor[:, :2] * cell_size + _tensor[:, 2:4] * .5 # compute x2, y2
            tensor_ltrb[:,4] = _tensor[:,4] # pass conf
            return tensor_ltrb

        # mask for the cells which contain object
        batch_size = target_tensor.shape[0]
        mask_obj = target_tensor[:, :, :, 4] == 1 # [batch_size, S, S]
        mask_obj = mask_obj.unsqueeze(-1).expand_as(target_tensor) # [batch_size, S, S, Bx5+C], 5=len([x, y, w, h, conf])
        # mask for the cells which does NOT contain object
        mask_noobj = target_tensor[:, :, :, 4] == 0 # [batch_size, S, S]
        mask_noobj = mask_noobj.unsqueeze(-1).expand_as(target_tensor) # [batch_size, S, S, Bx5+C], 5=len([x, y, w, h, conf])

        # pred_tensor which contain object: '(tensor)_bb' for bounding boxes, '(tensor)_class' for calsses
        pred_tensor_obj = pred_tensor[mask_obj].reshape([-1, self.B*5+self.C]) # [filtered, Bx5+C], 5=len([x, y, w, h, conf])
        pred_tensor_obj_bb = pred_tensor_obj[:, :self.B*5].reshape([-1, 5]) # [filtered x B, 5]
        pred_tensor_obj_class = pred_tensor_obj[:, self.B*5:].reshape([-1, self.C]) # [filtered, C]
        # target_tensor which contain object: '(tensor)_bb' for bounding boxes, '(tensor)_class' for calsses
        target_tensor_obj = target_tensor[mask_obj].reshape([-1, self.B*5+self.C]) # [filtered, Bx5+C], 5=len([x, y, w, h, conf])
        target_tensor_obj_bb = target_tensor_obj[:, :self.B*5].reshape([-1, 5]) # [filtered x B, 5]
        target_tensor_obj_class = target_tensor_obj[:, self.B*5:].reshape([-1, self.C]) # [filtered, C]

        # mask for the bounding boxes which is resposible for the ground truth.
        mask_resp = torch.ByteTensor(pred_tensor_obj_bb.size()).to(device)
        mask_resp.zero_()
        
        # gather iou for loss_obj
        target_tensor_obj_iou = torch.zeros_like(target_tensor_obj_bb).to(device)

        for i in range(0, target_tensor_obj_bb.shape[0], self.B):
            # preprocess
            pred_bb = pred_tensor_obj_bb[i:i+self.B] # [B, 5], 5=len([x, y, w, h, conf])
            target_bb = target_tensor_obj_bb[i:i+self.B]
            pred_bb_ltrb = center_to_ltrb(pred_bb) # [B, 5], 5=len([x1, y1, x2, y2, conf])
            target_bb_ltrb = center_to_ltrb(target_bb)

            # compute iou
            iou = self.compute_iou(pred_bb_ltrb[:, :4], target_bb_ltrb[0, :4].unsqueeze(0)) # [B, 1], target has duplicate ground truth as in encoder function in data.py
            bb_resp_iou, bb_resp_idx = iou.max(0) # choose maximum iou as resposible

            # update
            mask_resp[i+bb_resp_idx] = 1
            target_tensor_obj_iou[i+bb_resp_idx, 4] = bb_resp_iou.to(device)

        # --- compute each loss ---
        pred_resp = pred_tensor_obj_bb[mask_resp].reshape([-1, 5])
        target_resp = target_tensor_obj_bb[mask_resp].reshape([-1, 5])
        target_resp_iou = target_tensor_obj_iou[mask_resp].reshape([-1, 5]) # conf = P(Obj) * IOU(pred, truth)

        # 1. loss_xy
        loss_xy = torch.sum((target_resp[:, :2] - pred_resp[:, :2])**2) / batch_size

        # 2. loss_wh
        loss_wh = torch.sum((torch.sqrt(target_resp[:, 2:4]) - torch.sqrt(pred_resp[:, 2:4]))**2) / batch_size

        # 3. loss_obj
        loss_obj = torch.sum((target_resp_iou[:, 4] - pred_resp[:, 4])**2) / batch_size

        # 4. loss_noobj
        # I decided to consider both 'conf' for this loss, as discussed with TA in http://klms.kaist.ac.kr/mod/ubboard/article.php?id=352893&bwid=192633
        # pred_tensor & target_tensor which does NOT contain object
        pred_tensor_noobj = pred_tensor[mask_noobj].reshape([-1, self.B*5+self.C]) # [filtered, Bx5+C], 5=len([x, y, w, h, conf])
        target_tensor_noobj = target_tensor[mask_noobj].reshape([-1, self.B*5+self.C])
        pred_noobj_conf = pred_tensor_noobj[:, [4, 9]].reshape([-1, 2]) # consider both 'conf'
        target_noobj_conf = target_tensor_noobj[:, [4, 9]].reshape([-1, 2])
        loss_noobj = torch.sum((target_noobj_conf - pred_noobj_conf)**2) / batch_size

        # 5. loss_class
        loss_class = torch.sum((target_tensor_obj_class - pred_tensor_obj_class)**2) / batch_size

        return loss_xy, loss_wh, loss_obj, loss_noobj, loss_class

compute_loss = Loss(grid_size, num_boxes, num_classes)


# # compute_iou testing
# def compute_iou(bbox1, bbox2):
#     """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
#     Use this function if necessary.

#     Args:
#         bbox1: (Tensor) bounding bboxes, sized [N, 4].
#         bbox2: (Tensor) bounding bboxes, sized [M, 4].
#     Returns:
#         (Tensor) IoU, sized [N, M].
#     """
#     N = bbox1.size(0)
#     M = bbox2.size(0)

#     # Compute left-top coordinate of the intersections
#     lt = torch.max(
#         bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
#         bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
#     )
#     # Conpute right-bottom coordinate of the intersections
#     rb = torch.min(
#         bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
#         bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
#     )
#     # Compute area of the intersections from the coordinates
#     wh = rb - lt   # width and height of the intersection, [N, M, 2]
#     wh[wh < 0] = 0 # clip at 0
#     inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

#     # Compute area of the bboxes
#     area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
#     area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
#     area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
#     area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

#     # Compute IoU from the areas
#     union = area1 + area2 - inter # [N, M, 2]
#     iou = inter / union           # [N, M, 2]

#     return iou

# b1 = torch.tensor([[[3.,3.,4.,5.],[5.,4.,3.,2.]], [[8.,1.,9.,-1.],[-1.,0.,1.,3.]], [[2.,1.,0.,-1.],[-1.,0.,1.,9.]]])
# torch.sum(b1)
# b1
# b1.shape
# b1.argmax()
# b1.argmax(0)
# b1.argmax(1)
# b1.argmax(2)
# mask = (b1 <= 2)
# b1[mask]
# b1m = b1[:,3] > 3
# b1m1 = b1m.unsqueeze(-1).expand_as(b1)
# b1m1 * b1
# b1[:, :2] = b1[:, :2] - b1[:, 2:4]*0.5
# b1.shape[0]
# b2 = torch.tensor([[1.5,1.5,2.5,2.5]])
# IoU = compute_iou(b1, b2)
# print(b1.size(), b2.size(), IoU.size())
# print(IoU)
# int(np.argmax(IoU))
# compute_iou testing

# Problem 3. Implement Train/Test Pipeline
from torch.utils.tensorboard import SummaryWriter
log_dir = "./logs"
writer = SummaryWriter(log_dir)
tb_log_freq = 5
import time

# Training & Testing.
model = model.to(device)
best_test_loss_final = np.inf
for epoch in range(1, max_epoch):
    start_time = time.time()
    # Learning rate scheduling
    if epoch in [50, 150]:
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if epoch < last_epoch:
        continue

    model.train()
    for i, (x, y) in enumerate(train_dloader):
        # implement training pipeline here
        # 1. set proper device
        x = x.to(device) # torch.Size([64, 3, 224, 224])
        y = y.to(device) # torch.Size([64, 7, 7, 30])

        # 2. feed and get output from network
        y_pred = model(x)

        # 2. compute loss aggregated to a single final loss as paper
        loss_function = Loss()
        loss_xy, loss_wh, loss_obj, loss_noobj, loss_class = loss_function(y_pred, y)
        train_loss_final = lambda_coord*(loss_xy+loss_wh) + loss_obj + lambda_noobj*loss_noobj + loss_class

        # 3. backward and update
        optimizer.zero_grad()
        train_loss_final.backward()
        optimizer.step()

        # tensorboard
        n_iter = epoch * len(train_dloader) + i
        if n_iter % tb_log_freq == 0:
            writer.add_scalar('train/loss', train_loss_final, n_iter)

    model.eval()
    with torch.no_grad():
        for x, y in test_dloader:
            # implement testing pipeline here
            # 1. set proper device
            x = x.to(device) # torch.Size([64, 3, 224, 224])
            y = y.to(device) # torch.Size([64, 7, 7, 30])

            # 2. feed and get output from network
            y_pred = model(x)

            # 2. compute loss aggregated to a single final loss as paper
            loss_function = Loss()
            loss_xy, loss_wh, loss_obj, loss_noobj, loss_class = loss_function(y_pred, y)
            test_loss_final = lambda_coord*(loss_xy+loss_wh) + loss_obj + lambda_noobj*loss_noobj + loss_class

            # tensorboard
            n_iter = epoch * len(test_dloader) + i
            if n_iter % tb_log_freq == 0:
                writer.add_scalar('test/loss', test_loss_final, n_iter)
    
    if test_loss_final < best_test_loss_final:
        best_test_loss_final = test_loss_final

    # save the results
    ckpt = {'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch}
    torch.save(ckpt, ckpt_path)

    # print
    print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f, Time: %.4f'
    % (epoch + 1, max_epoch, test_loss_final, best_test_loss_final, time.time() - start_time))


# Problem 4. Implement decoder to extract bounding boxes from output-grids
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def NMS(bboxes, scores, threshold=0.35):
    ''' Non Max Suppression
    Args:
        bboxes: (torch.tensors) list of bounding boxes. size:(N, 4) ((left_top_x, left_top_y, right_bottom_x, right_bottom_y), (...))
        probs: (torch.tensors) list of confidence probability. size:(N,) 
        threshold: (float)   
    Returns:
        keep_dim: (torch.tensors)
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        try:
            i = order[0]
        except:
            i = order.item()
        keep.append(i)

        if order.numel() == 1: break

        xx1 = x1[order[1:]].clamp(min=x1[i]) # consider intersection
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0) # clamp out which is outbounded from current box
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter) # iou
        ids = (ovr <= threshold).nonzero().squeeze() # leave only over the threshold
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    keep_dim = torch.LongTensor(keep)
    return keep_dim

def inference(model, image_path):
    """ Inference function
    Args:
        model: (nn.Module) Trained YOLO model.
        image_path: (str) Path for loading the image.
    """
    # load & pre-processing
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)

    h, w, c = image.shape
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = transform(torch.from_numpy(img).float().div(255).transpose(2, 1).transpose(1, 0)) #Normalization
    img = img.unsqueeze(0)
    img = img.to(device)

    # inference
    output_grid = model(img).cpu()

    #### YOU SHOULD IMPLEMENT FOLLOWING decoder FUNCTION ####
    # decode the output grid to the detected bounding boxes, classes and probabilities.
    bboxes, class_idxs, probs = decoder(output_grid)
    num_bboxes = bboxes.size(0)

    # draw bounding boxes & class name
    for i in range(num_bboxes):
        bbox = bboxes[i]
        class_name = VOC_CLASSES[class_idxs[i]]
        prob = probs[i]

        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, '%s: %.2f'%(class_name, prob), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1,
                    8)

    cv2.imwrite(image_name.replace('.jpg', '_result.jpg'), image)

def decoder(grid):
    """ Decoder function that decode the output-grid to bounding box, class and probability. 
    Args:
        grid: (torch.tensors)
    Returns:
        bboxes: (torch.tensors) list of bounding boxes. size:(N, 4) ((left_top_x, left_top_y, right_bottom_x, right_bottom_y), (...))
        class_idxs: (torch.tensors) list of class index. size:(N,)
        probs: (torch.tensors) list of confidence probability. size:(N,)
    """

    grid_num = 7
    bboxes = []
    class_idxs = []
    probs = []

    S = grid_num
    B = 2
    C = 20

    def rel_center_to_abs_ltrb(_tensor):
        """ Transform tensor from relative 'center' to absolute 'left-top, right-bottom' normalized in image-size.

        Args:
            _tensor (Tensor) : original, sized [S x S x B, 5], 5=len([x, y, w, h, conf]).
        Returns:
            tensor_abs_ltrb (Tensor) : sized [S x S x B, 5] where we have 'filtered' cells vary in context, 5=len([x1, y1, x2, y2, conf]) normalized in image-size.
        """
        # As in encoder function, both w,h are in image-size and both x,y are in cell-size
        cell_size = 1./S

        # transform xy coordinates to be normalized in image-size.
        tensor_ltrb = torch.zeros_like(_tensor)
        for i in range(0, _tensor.shape[0], B):
            cell_x0 = int(i/B) % S
            cell_y0 = int(i/B) // S # j
            # print("cell_y0x0:", i, (cell_y0, cell_x0), _tensor[i])
            # print("cell_y0x02:", i, (cell_y0, cell_x0), _tensor[i+1])
            abs_cell_x0y0 = torch.FloatTensor([cell_x0, cell_y0]) * cell_size # as a reverse of encoder, the left-top coordinates of this cell normalized in image-size
            # compute x1, y1, x2, y2 normalized in image-size / w,h are already normalized in image-size
            tensor_ltrb[i:i+2, :2] = _tensor[i:i+2, :2] * cell_size + abs_cell_x0y0 - 0.5 * _tensor[i:i+2, 2:4]
            tensor_ltrb[i:i+2, 2:4] = _tensor[i:i+2, :2] * cell_size + abs_cell_x0y0 + 0.5 * _tensor[i:i+2, 2:4] 
            tensor_ltrb[i:i+2, 4] = _tensor[i:i+2, 4] # pass conf
        
        return tensor_ltrb # abs_ltrb

    # for i in range(S):
    #     for j in range(S):
    #         print("ji:", (j,i))
    #         print(grid[j, i, :5])
    #         print(grid[j, i, 5:5*B])

    # grid : [S, S, Bx5+C] = [7, 7, 30]
    grid = grid.squeeze().cpu().data
    
    # extract coordinates
    grid_coord = grid[:,:,:B*5].reshape([-1, 5]) # [S x S x B, 5], 5=len([x, y, w, h, conf])
    grid_coord_ltrb = rel_center_to_abs_ltrb(grid_coord) # [S x S x B, 5], 5=len([x1, y1, x2, y2, conf]) in real image-size

    # making class indices
    grid_class = grid[:,:,B*5:] # [S, S, C]
    grid_class = grid_class.repeat([1,1,B]).reshape([-1, C]) # [S x S x B, C]

    # # compute class scores for filtering out lower oness
    # score_threshold=0.2
    # class_scores = grid_class * grid_coord_ltrb[:,4].unsqueeze(-1) # [S x S x B, C]
    # class_scores[class_scores < score_threshold] = 0 # set zero if score < score_threshold

    # get class indicies
    grid_class_scores, grid_class_idxs = torch.max(grid_class, -1) # choose the highest score as final prediction
    
    # update results
    bboxes.append(grid_coord_ltrb[:,:4]) # [S X S X B, 4], we only need (left_top_x, left_top_y, right_bottom_x, right_bottom_y) for each boxes
    probs.append(grid_class_scores * grid_coord_ltrb[:,4]) # [S X S X B]
    class_idxs.append(grid_class_idxs) # [S X S X B]

    if len(bboxes) == 0: # Any box was not detected
        bboxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        class_idxs = torch.zeros(1)
        
    else: 
        #list of tensors -> tensors
        bboxes = torch.stack(bboxes).squeeze()
        probs = torch.stack(probs).squeeze()
        class_idxs = torch.stack(class_idxs).squeeze()

    keep_dim = NMS(bboxes, probs, threshold=0.35) # Non Max Suppression

    # select bboxes to draw
    bboxes_nms, class_nms, probs_nms = bboxes[keep_dim], class_idxs[keep_dim], probs[keep_dim]
    mask_prob = (probs_nms > 0.1) # i choosed the threshold as 0.1 here, but it could be different
    probs_nms = probs_nms[mask_prob]
    class_nms = class_nms[mask_prob]
    mask_prob = mask_prob.unsqueeze(-1).expand_as(bboxes_nms)
    bboxes_nms = bboxes_nms[mask_prob].reshape([-1, 4])

    return bboxes_nms, class_nms, probs_nms

test_image_dir = 'test_images'
image_path_list = [os.path.join(test_image_dir, path) for path in os.listdir(test_image_dir)]

for image_path in image_path_list:
    inference(model, image_path)