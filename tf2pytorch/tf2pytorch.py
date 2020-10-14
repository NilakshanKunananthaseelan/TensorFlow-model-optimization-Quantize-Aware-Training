 

import h5py
import numpy as np
import matplotlib.pyplot  as plt

import torch
import torch.nn as nn

from model_PT import PT_ResNet50




 

f = h5py.File('output/saved_model/resnet50.h5')

mem = [i for i in f['model_weights']]

model = PT_ResNet50.resnet50()


model_params = {}
 
for i,ele in enumerate(mem):
 
    e = ele
    grp = f['model_weights'][ele]
    grp_type = [i for i in grp]  
    if(len(grp_type)>0):
        conv_grp = f['model_weights'][ele][grp_type[0]]
        param = [i for i in conv_grp]
 
        for i in range (len(param)):
#                 print("{} : {}".format(grp_type[0]+'_'+param[i],conv_grp[param[i]].value.shape))
                model_params[ele+'_'+param[i]] = conv_grp[param[i]].value

 
map_dict = {'layer1':'conv2',
            'layer2':'conv3',
            'layer3':'conv4',
            'layer4':'conv5',
             '0':'block1',
             '1':'block2',
             '2':'block3',
             '3':'block4',
             '4':'block5',
             '5':'block6',
             'conv1':'1_conv',
             'conv2':'2_conv',
             'conv3':'3_conv',
             'conv4':'4_conv',
             'conv5':'5_conv',
             'bn1' : '1_bn',
             'bn2':'2_bn',
             'bn3':'3_bn',
             'bn4':'4_bn',
             'downsample':'0_conv',
             'bn_downsample':'0_bn_',         
              
             'bn_weight':'gamma:0',
             'bn_bias':'beta:0',
             'bn_running_mean':'moving_mean:0',
             'bn_running_var':'moving_variance:0',
             'weight':'kernel:0',
             'bias':'bias:0',
             'fc.weight':'predictions_kernel:0',
             'fc.bias':'predictions_bias:0'}

  



layers = [k for k in model.state_dict().keys() if 'tracked' not in k]

t2p = []
for c,layer in enumerate(layers):
 
    l = layer.split('.')
    
    tf_name = ''
    print(layer)
    if 'conv' in layer or (l[-2] == '0' and (l[-1]=='weight' or l[-1]=='bias')):
#         
        if(len(l)>2):
            for i,ele in enumerate(l):
                 
                if(i==len(l)-2) and ele=='0':pass
                else:
                    
                    tf_name+=map_dict[ele]
                    if(i<len(l)-1):
                        tf_name+='_'
                
        else:
            tf_name+=l[0]+'_conv_'+map_dict[l[-1]]
    elif 'bn' in layer or (l[-2] == '1'):
        
        if(len(l))>2:
            for i,ele in enumerate(l):
                if i<len(l)-1  and ele !='downsample':
                    if(i==len(l)-2) and ele=='1':pass
                    else:
                        tf_name+=map_dict[ele]+'_'
                
                else:
                    
                    tf_name+=map_dict['bn_'+ele]
                    
        else:
            tf_name='conv1_bn_'+map_dict['bn_'+l[-1]]


 
    try:
        print(" Py_layer: {} {} \n tf_layer: {} {} \n".format(layer,model.state_dict()[layer].shape,tf_name,model_params[tf_name].T.shape))
        if('kernel' in tf_name):
             model.state_dict()[layer][:] = torch.from_numpy(np.transpose(model_params[tf_name], (3, 2, 0, 1)))
        else:
             model.state_dict()[layer][:] = torch.from_numpy((model_params[tf_name]))

    except:
        tf_name = map_dict[layer]
        print(" Py_layer: {} {} \n tf_layer: {} {} \n".format(layer,model.state_dict()[layer].shape,tf_name,model_params[tf_name].T.shape))
        if('kernel' in tf_name):
              model.state_dict()[layer][:] = torch.from_numpy(np.transpose(model_params[tf_name],(1,0)))
        else:
              model.state_dict()[layer][:] = torch.from_numpy((model_params[tf_name]))

    t2p.append(tf_name)
        
        
print(c)

 

# for k in model_params.keys():
#     if(k not in t2p):
#         print(k)
#     else:
#         print("**",k)


# # In[16]:
from collections import OrderedDict
weight_dict= OrderedDict([('module.'+k, v)  for k, v in model.state_dict().items()])



path = 'tf2pytorch_fp.pth.tar'
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.6,nesterov=True)

torch.save({'state_dict':weight_dict,
            # 'loss':0.44410261511802673, #set loss of the trained model
            # 'epoch':4,
            # 'optimizer_state_dict':optimizer.state_dict(), 
            # 'optimizer_type':type(optimizer) #check optimizer
            },path)

for k,v in model.state_dict().items():
    if('conv' in k):
        print("{} : {} ".format(k,v.shape))

        
        w_p = [i for i in np.array(v).flatten() if i >0 ]
        print("minimum positive value :{}".format(min(w_p)))


#-------------EVALUATE-PyTorch RESNET50---------------------------------------------

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
import random
import shutil
import time
import warnings




data_path = '/home/ubuntu/work/data.imagenet_nilakshan/'


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) #TF doesn't do this

valdir = os.path.join(data_path, 'val') #test data
 
criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                 momentum=0.5,nesterov=True)

checkpoint=torch.load(path)
model = model.cuda()
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir,transforms.Compose([
            transforms.Resize((256)),
            transforms.CenterCrop(224),
            transforms.ToTensor()#   normalize,
        ])),
        batch_size=32, shuffle=False,
        num_workers=1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def validate(val_loader, model, criterion,):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
             
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
             
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 60 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




validate(val_loader, model, criterion)

