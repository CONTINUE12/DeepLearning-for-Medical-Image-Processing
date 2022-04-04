import numpy as np
import pandas as pd
import os
import os.path
fold = 0#1#4#3
resep = 962#21#17#39
gbtdepth = 2#3#2#3
neptime = 0.3
testdetp = -1.5
traindetp = -1.5
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
resmodelpath = './detcls-'+str(fold)+'-old/ckptgbt.t7'
def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)): overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union
def nms(output, nms_th):
    if len(output) == 0: return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1: bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes
# find the mapping
# load groundtruth
antclscsv = pd.read_csv('/home/zhaojie/zhaojie/Lung/code/nodcls/data/annotationdetclsconvfnl_v3.csv', \
    names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])#1004
srslst = antclscsv['seriesuid'].tolist()[1:]
cdxlst = antclscsv['coordX'].tolist()[1:]
cdylst = antclscsv['coordY'].tolist()[1:]
cdzlst = antclscsv['coordZ'].tolist()[1:]
dimlst = antclscsv['diameter_mm'].tolist()[1:]
mlglst = antclscsv['malignant'].tolist()[1:]
gtdct = {}
for idx in range(len(srslst)):
    vlu = [float(cdxlst[idx]), float(cdylst[idx]), float(cdzlst[idx]), float(dimlst[idx]), int(mlglst[idx])]
    if srslst[idx].split('-')[0] not in gtdct: gtdct[srslst[idx].split('-')[0]] = [vlu]
    else: gtdct[srslst[idx].split('-')[0]].append(vlu)
# print('len(srslst)',len(srslst))#1004 在annotationdetclsconvfnl_v3
# print('len(gtdct)',len(gtdct))#554不重复的个数
tedetpath = '/home/zhaojie/zhaojie/Lung/code/detector_py3/results/dpn3d26/Test_Pbb/Test6789-predanno'+str(testdetp)+'.csv'
# fid = open(tedetpath, 'r')#fold = 0
prdcsv = pd.read_csv(tedetpath, names=['seriesuid','coordX','coordY','coordZ','probability'])
srslst = prdcsv['seriesuid'].tolist()[1:]
cdxlst = prdcsv['coordX'].tolist()[1:]
cdylst = prdcsv['coordY'].tolist()[1:]
cdzlst = prdcsv['coordZ'].tolist()[1:]
prblst = prdcsv['probability'].tolist()[1:]
# build dict first for rach seriesuid
srsdct = {}
for idx in range(len(srslst)):
    vlu = [cdxlst[idx], cdylst[idx], cdzlst[idx], prblst[idx]]
    if srslst[idx] not in srsdct: srsdct[srslst[idx]] = [vlu]
    else: srsdct[srslst[idx]].append(vlu)
# print('len(srslst)',len(srslst))#778#检测到的
# print('len(gtdct)',len(srsdct))#88=一个fold0的数量
# pbb path, find the mapping of csv to pbb
pbbpth = '/home/zhaojie/zhaojie/Lung/code/detector_py3/results/dpn3d26/'+'Test_Pbb/'
# rawpth = '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/'
# prppth = '/home/zhaojie/zhaojie/Lung/data/luna16/LUNA16PROPOCESSPATH/'
trudat = {}
tefnmlst = []
tecdxlst = []
tecdylst = []
tecdzlst = []
telablst = []
tedimlst = []
import math
for srs, vlu in srsdct.items():
    pbb = np.load(os.path.join(pbbpth, srs+'_pbb.npy'))
    lbb = np.load(os.path.join(pbbpth, srs+'_lbb.npy')) # list, x y z d
    # sliceim,origin,spacing,isflip = load_itk_image(os.path.join(rawpth, srslst[idx]+'.mhd'))
 #    origin = np.load(os.path.join(prppth, srslst[idx]+'_origin.npy'))
 #    spacing = np.load(os.path.join(prppth, srslst[idx]+'_spacing.npy'))
 #    resolution = np.array([1, 1, 1])
 #    extendbox = np.load(os.path.join(prppth, srslst[idx]+'_extendbox.npy'))
    # print('pbb',pbb)#shape(36, 5)
    pbbold = np.array(pbb[pbb[:,0] > testdetp])#detp])
    pbb = nms(pbbold, 0.1)
    # print('0',pbb.shape, len(vlu))#(10, 5) 10不固定
    assert pbb.shape[0] == len(vlu)
    kptpbb = np.array(pbb)#[:5, :]) # prob, x, y, z, d
    # print('kptpbb',kptpbb)
    # find the true label
    for idx in range(kptpbb.shape[0]):
        tefnmlst.append(srs)
        tecdxlst.append(kptpbb[idx, 1])
        tecdylst.append(kptpbb[idx, 2])
        tecdzlst.append(kptpbb[idx, 3])
        tedimlst.append(kptpbb[idx, 4])
        if lbb.shape[0] == 0 or (lbb.shape[0]==1 and abs(lbb[0,0])+abs(lbb[0,1])+abs(lbb[0,2])+abs(lbb[0,3])==0):
            kptpbb[idx, 0] = 0
            telablst.append(0)
            continue
        ispos = 0
        if srs in gtdct:
            for l in gtdct[srs]:
                if math.pow(l[0]-kptpbb[idx,1],2.) + math.pow(l[1]-kptpbb[idx,2],2.) + math.pow(l[2]-kptpbb[idx,3],2.) < \
                  math.pow(max(16., l[3]/2),2.):
                    kptpbb[idx, 0] = l[4]
                    telablst.append(l[4])
                    ispos = 1
                    break
        if ispos == 0: 
            kptpbb[idx, 0] = 0
            telablst.append(0)
        trudat[srs] = kptpbb
print('test',len(telablst), sum(telablst), np.sum(kptpbb[:,0]))#778检测到的 50 0.0

# load train data
tedetpath = '/home/zhaojie/zhaojie/Lung/code/detector_py3/results/dpn3d26/Train_Pbb/Train012345-predanno'+str(traindetp)+'.csv'
# fid = open(tedetpath, 'r')
prdcsv = pd.read_csv(tedetpath, names=['seriesuid','coordX','coordY','coordZ','probability'])
srslst = prdcsv['seriesuid'].tolist()[1:]
cdxlst = prdcsv['coordX'].tolist()[1:]
cdylst = prdcsv['coordY'].tolist()[1:]
cdzlst = prdcsv['coordZ'].tolist()[1:]
prblst = prdcsv['probability'].tolist()[1:]
# build dict first for rach seriesuid
srsdct = {}
for idx in range(len(srslst)):
    vlu = [cdxlst[idx], cdylst[idx], cdzlst[idx], prblst[idx]]
    if srslst[idx] not in srsdct: srsdct[srslst[idx]] = [vlu]
    else: srsdct[srslst[idx]].append(vlu)
# print('0',len(srsdct['1.3.6.1.4.1.14519.5.2.1.6279.6001.265960756233787099041040311282']))
# pbb path, find the mapping of csv to pbb
pbbpth = '/home/zhaojie/zhaojie/Lung/code/detector_py3/results/dpn3d26/Train_Pbb/'
# rawpth = '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/'
# prppth = '/home/zhaojie/zhaojie/Lung/data/luna16/LUNA16PROPOCESSPATH/'
trudat = {}
trfnmlst = []
trcdxlst = []
trcdylst = []
trcdzlst = []
trlablst = []
trdimlst = []
import math
for srs, vlu in srsdct.items():
    if 1 == 1:
    # if str(srs) == '1.3.6.1.4.1.14519.5.2.1.6279.6001.265960756233787099041040311282':
        # print(os.path.join(pbbpth, srs+'_pbb.npy')) 
        pbb = np.load(os.path.join(pbbpth, srs+'_pbb.npy'),mmap_mode='r')
        # pbb = np.load('/home/zhaojie/zhaojie/Lung/code/detector_py3/results/dpn3d26/retrft965/val995/1.3.6.1.4.1.14519.5.2.1.6279.6001.265960756233787099041040311282_pbb.npy',mmap_mode='r')
        # print('pbb.shape',pbb.shape)#(
        lbb = np.load(os.path.join(pbbpth, srs+'_lbb.npy'))
        # sliceim,origin,spacing,isflip = load_itk_image(os.path.join(rawpth, srslst[idx]+'.mhd'))
        #    origin = np.load(os.path.join(prppth, srslst[idx]+'_origin.npy'))
        #    spacing = np.load(os.path.join(prppth, srslst[idx]+'_spacing.npy'))
        #    resolution = np.array([1, 1, 1])
        #    extendbox = np.load(os.path.join(prppth, srslst[idx]+'_extendbox.npy'))
        pbbold = np.array(pbb[pbb[:,0] > traindetp])#detp])
        pbbold = np.array(pbbold[pbbold[:,-1] > 3])  # add new 9 15#根据半径过滤掉小于3mm的
        pbbold = pbbold[np.argsort(-pbbold[:,0])][:1000] 
        pbb = nms(pbbold, 0.1)
        # print(pbbold.shape,pbb.shape, len(vlu))
        assert pbb.shape[0] == len(vlu)
        kptpbb = np.array(pbb)#pbb[:5, :]) # prob, x, y, z, d # :5 is the first version
        # find the true label
        for idx in range(kptpbb.shape[0]):
            trfnmlst.append(srs)
            trcdxlst.append(kptpbb[idx, 1])
            trcdylst.append(kptpbb[idx, 2])
            trcdzlst.append(kptpbb[idx, 3])
            trdimlst.append(kptpbb[idx, 4])
            if lbb.shape[0] == 0 or (lbb.shape[0]==1 and abs(lbb[0,0])+abs(lbb[0,1])+abs(lbb[0,2])+abs(lbb[0,3])==0):
                kptpbb[idx, 0] = 0
                trlablst.append(0)
                continue
            ispos = 0
            if srs in gtdct:
                for l in gtdct[srs]:
                    if math.pow(l[0]-kptpbb[idx,1],2.) + math.pow(l[1]-kptpbb[idx,2],2.) + math.pow(l[2]-kptpbb[idx,3],2.) < \
                      math.pow(max(16., l[3]/2),2.):
                        kptpbb[idx, 0] = l[4]
                        trlablst.append(l[4])
                        ispos = 1
                        break
            if ispos == 0: 
                kptpbb[idx, 0] = 0
                trlablst.append(0)
            trudat[srs] = kptpbb
print('train',len(trlablst), sum(trlablst), np.sum(kptpbb[:,0]))#632 48 0.0

# save the data - later
# run test
import numpy as np
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from models import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import os
import argparse
from models.dpn3d import DPN92_3D
from utils import progress_bar
from torch.autograd import Variable
import numpy as np
criterion = nn.CrossEntropyLoss()
CROPSIZE = 17
blklst = []
# blklst = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286-388', \
#     '1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286-389', \
#     '1.3.6.1.4.1.14519.5.2.1.6279.6001.132817748896065918417924920957-660']
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
best_acc_gbt = 0
# start_epoch = 50  # start from epoch 0 or last checkpoint epoch
# Cal mean std
preprocesspath = '/home/zhaojie/zhaojie/Lung/data/luna16/cls/crop_v3/'
preprocessalltrainpath = '/home/zhaojie/zhaojie/Lung/data/luna16/LUNA16PROPOCESSPATH/Train_npy012345/'
preprocessalltestpath = '/home/zhaojie/zhaojie/Lung/data/luna16/LUNA16PROPOCESSPATH/Test_npy6789/'
pixvlu, npix = 0, 0
for fname in os.listdir(preprocesspath):
    if fname.endswith('.npy'):
        if fname[:-4] in blklst: continue
        data = np.load(os.path.join(preprocesspath, fname))
        pixvlu += np.sum(data)
        npix += np.prod(data.shape)
pixmean = pixvlu / float(npix)
pixvlu = 0
for fname in os.listdir(preprocesspath):
    if fname.endswith('.npy'):
        if fname[:-4] in blklst: continue
        data = np.load(os.path.join(preprocesspath, fname))-pixmean
        pixvlu += np.sum(data * data)
pixstd = np.sqrt(pixvlu / float(npix))
# pixstd /= 255
print(pixmean, pixstd)
print('mean '+str(pixmean)+' std '+str(pixstd))
# Datatransforms
print('==> Preparing data..') # Random Crop, Zero out, x z flip, scale, 
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),
])
transform_train = transforms.Compose([ 
    # transforms.RandomScale(range(28, 38)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomYFlip(),
    transforms.RandomZFlip(),
    transforms.ZeroOut(4),
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)), # need to cal mean and std, revise norm func
])
from dataloadernp import lunanod
import pandas as pd
import logging
# fold = 1
# gbtdepth = 3
savemodelpath = './detcls-'+str(fold)+'new/'
if not os.path.isdir(savemodelpath):
    os.mkdir(savemodelpath)
logging.basicConfig(filename=savemodelpath+'detclslog-'+str(fold), level=logging.INFO)
mxx = mxy = mxz = mxd = 0
tefnamelst = []
telabellst = []
tefeatlst = []
trfnamelst = []
trlabellst = []
trfeatlst = []
for srsid, label, x, y, z, d in zip(tefnmlst, telablst, tecdxlst, tecdylst, tecdzlst, tedimlst):
    mxx = max(abs(float(x)), mxx)
    mxy = max(abs(float(y)), mxy)
    mxz = max(abs(float(z)), mxz)
    mxd = max(abs(float(d)), mxd)
    if srsid in blklst: continue
    # crop raw pixel as feature
    data = np.load(os.path.join(preprocessalltestpath, srsid+'_clean.npy'))#'/home/zhaojie/zhaojie/Lung/data/luna16/LUNA16PROPOCESSPATH/subset9/'
    # print data.shape
    bgx = int(min(data.shape[1],max(0,x-CROPSIZE/2)))
    bgy = int(min(data.shape[2],max(0,y-CROPSIZE/2)))
    bgz = int(min(data.shape[3],max(0,z-CROPSIZE/2)))
    data0 = np.array(data[0,bgx:bgx+CROPSIZE, bgy:bgy+CROPSIZE, bgz:bgz+CROPSIZE])
    # print data0.shape
    data1 = np.ones((CROPSIZE, CROPSIZE, CROPSIZE)) * 170
    data1[:data0.shape[0], :data0.shape[1], :data0.shape[2]] = np.array(data0)
    # print data1.shape
    feat = np.hstack((np.reshape(data1, (-1,)) / 255, float(d)))
    # if srsid.split('-')[0] in teidlst:

    bgx = int(min(data.shape[1],max(0,x-32/2)))
    bgy = int(min(data.shape[2],max(0,y-32/2)))
    bgz = int(min(data.shape[3],max(0,z-32/2)))
    data0 = np.array(data[0,bgx:bgx+32, bgy:bgy+32, bgz:bgz+32])
    # print data0.shape
    data1 = np.ones((32, 32, 32)) * 170
    data1[:data0.shape[0], :data0.shape[1], :data0.shape[2]] = np.array(data0)

    tefnamelst.append(data1)
    telabellst.append(int(label))
    tefeatlst.append(feat)
print('len(telabellst), sum(telabellst)',len(telabellst), sum(telabellst))
for srsid, label, x, y, z, d in zip(trfnmlst, trlablst, trcdxlst, trcdylst, trcdzlst, trdimlst):
    mxx = max(abs(float(x)), mxx)
    mxy = max(abs(float(y)), mxy)
    mxz = max(abs(float(z)), mxz)
    mxd = max(abs(float(d)), mxd)
    if srsid in blklst: continue
    # crop raw pixel as feature
    data = np.load(os.path.join(preprocessalltrainpath, srsid+'_clean.npy'))
    # print data.shape
    bgx = int(min(data.shape[1],max(0,x-CROPSIZE/2)))
    bgy = int(min(data.shape[2],max(0,y-CROPSIZE/2)))
    bgz = int(min(data.shape[3],max(0,z-CROPSIZE/2)))
    data0 = np.array(data[0,bgx:bgx+CROPSIZE, bgy:bgy+CROPSIZE, bgz:bgz+CROPSIZE])
    # print data0.shape
    data1 = np.ones((CROPSIZE, CROPSIZE, CROPSIZE)) * 170
    data1[:data0.shape[0], :data0.shape[1], :data0.shape[2]] = np.array(data0)
    # print data1.shape
    feat = np.hstack((np.reshape(data1, (-1,)) / 255, float(d)))
    # if srsid.split('-')[0] in teidlst:

    bgx = int(min(data.shape[1],max(0,x-32/2)))
    bgy = int(min(data.shape[2],max(0,y-32/2)))
    bgz = int(min(data.shape[3],max(0,z-32/2)))
    data0 = np.array(data[0,bgx:bgx+32, bgy:bgy+32, bgz:bgz+32])
    # print data0.shape
    data1 = np.ones((32, 32, 32)) * 170
    data1[:data0.shape[0], :data0.shape[1], :data0.shape[2]] = np.array(data0)

    trfnamelst.append(data1)
    trlabellst.append(int(label))
    trfeatlst.append(feat)
print('len(trlabellst), sum(trlabellst)',len(trlabellst), sum(trlabellst))
for idx in range(len(trfeatlst)):
    # trfeatlst[idx][0] /= mxx
    # trfeatlst[idx][1] /= mxy
    # trfeatlst[idx][2] /= mxz
    trfeatlst[idx][-1] /= mxd
for idx in range(len(tefeatlst)):
    # tefeatlst[idx][0] /= mxx
    # tefeatlst[idx][1] /= mxy
    # tefeatlst[idx][2] /= mxz
    tefeatlst[idx][-1] /= mxd
# trainset = lunanod(trfnamelst, trlabellst, trfeatlst, train=False, transform=transform_test)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=30)
print(len(tefnamelst), sum(telablst), len(trfnamelst), sum(trlablst))
trainset = lunanod(preprocessalltrainpath, trfnamelst, trlabellst, trfeatlst, train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=30)
testset = lunanod(preprocessalltestpath, tefnamelst, telabellst, tefeatlst, train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=30)
print("Train: %d" %len(trainloader.dataset))#912
print("Val: %d" %len(testloader.dataset))#92
inputs, targets, feat = next(iter(trainloader))
print("Inputs: ", inputs.size())#torch.Size([16, 1, 32, 32, 32])
print("Targets: ", targets.size())#torch.Size([16])
print("feat: ", feat.size())#torch.Size([16, 4914])
# print(stop)
# checkpoint = torch.load(resmodelpath)#'./checkpoint-1-45/ckpt.t7')
# print(checkpoint.keys())
net = DPN92_3D()
# net = checkpoint['net']
# neptime = 0.2
def get_lr(epoch):
    if epoch < 150*neptime:
        lr = 0.1 #args.lr
    elif epoch < 300*neptime:
        lr = 0.01
    else:
        lr = 0.001
    return lr
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = False #True
import pickle
from sklearn.ensemble import GradientBoostingClassifier as gbt
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
def train(epoch):
    logging.info('\nEpoch: '+str(epoch))
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    train_loss = 0
    correct = 0
    total = 0
    trainfeat = np.zeros((len(trfnamelst), 2560+CROPSIZE*CROPSIZE*CROPSIZE+1))
    trainlabel = np.zeros((len(trfnamelst),))
    idx = 0
    for batch_idx, (inputs, targets, feat) in enumerate(trainloader):
        if use_cuda:
            # print(len(inputs), len(targets), len(feat), type(inputs[0]), type(targets[0]), type(feat[0]))
            # print(type(targets), type(inputs), len(targets))
            # targetarr = np.zeros((len(targets),))
            # for idx in xrange(len(targets)):
                # targetarr[idx] = targets[idx]
            # print((Variable(torch.from_numpy(targetarr)).data).cpu().numpy().shape)
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        outputs, dfeat = net(inputs) 
        # add feature into the array
        # print(torch.stack(targets).data.numpy().shape, torch.stack(feat).data.numpy().shape)
        # print((dfeat.data).cpu().numpy().shape)
        trainfeat[idx:idx+len(targets), :2560] = np.array((dfeat.data).cpu().numpy())
        for i in range(len(targets)):
            trainfeat[idx+i, 2560:] = np.array((Variable(feat[i]).data).cpu().numpy())
            trainlabel[idx+i] = np.array((targets[i].data).cpu().numpy())
        idx += len(targets)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    accout = round(correct.data.cpu().numpy() / total, 4)
    print('TrainLoss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx + 1), 100.*accout, correct, total))
    
    m = gbt(max_depth=gbtdepth, random_state=0)
    m.fit(trainfeat, trainlabel)
    gbttracc = round(np.mean(m.predict(trainfeat) == trainlabel),4)
    print('ep '+str(epoch)+' tracc '+str(accout)+' lr '+str(lr)+' gbtacc '+str(gbttracc))
    logging.info('ep '+str(epoch)+' tracc '+str(accout)+' lr '+str(lr)+' gbtacc '+str(gbttracc))
    return m

def test(epoch, m):
    global best_acc
    global best_acc_gbt
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    testfeat = np.zeros((len(tefnamelst), 2560+CROPSIZE*CROPSIZE*CROPSIZE+1))
    testlabel = np.zeros((len(tefnamelst),))
    dpnpred = np.zeros((len(tefnamelst),))
    idx = 0
    for batch_idx, (inputs, targets, feat) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, dfeat = net(inputs)
        # add feature into the array
        testfeat[idx:idx+len(targets), :2560] = np.array((dfeat.data).cpu().numpy())
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            # % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        for i in range(len(targets)):
            testfeat[idx+i, 2560:] = np.array((Variable(feat[i]).data).cpu().numpy())
            testlabel[idx+i] = np.array((targets[i].data).cpu().numpy())
            dpnpred[idx+i] = np.array((Variable(predicted[i]).data).cpu().numpy())
        idx += len(targets)
    # print(testlabel.shape, testfeat.shape, testlabel)#, trainfeat[:, 3])
    accout = round(correct.data.cpu().numpy() / total, 4)
    print('ValLoss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*accout, correct, total))
    gbtpred = m.predict(testfeat)
    np.save(savemodelpath+'gbtpred'+str(epoch)+'.npy', gbtpred)
    np.save(savemodelpath+'dpnpred'+str(epoch)+'.npy', dpnpred)
    gbtteacc = round(np.mean(gbtpred == testlabel),4)
    if gbtteacc > best_acc_gbt:
        if not os.path.isdir(savemodelpath):
            os.mkdir(savemodelpath)
        pickle.dump(m, open(savemodelpath+'gbtmodel-'+str(fold)+'.sav', 'wb'))
        logging.info('Saving gbt ..')
        state = {
            'net': net.module if use_cuda else net,
            'epoch': epoch,
        }
        if not os.path.isdir(savemodelpath):
            os.mkdir(savemodelpath)
        torch.save(state, savemodelpath + str(epoch) + '_' + str(gbtteacc) + '_ckptgbt.t7')
        best_acc_gbt = gbtteacc
    # Save checkpoint.
    acc = accout
    if acc > best_acc:
        logging.info('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(savemodelpath):
            os.mkdir(savemodelpath)
        torch.save(state, savemodelpath+'ckpt.t7')
        best_acc = acc
    logging.info('Saving..')
    state = {
        'net': net.module if use_cuda else net,
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir(savemodelpath):
        os.mkdir(savemodelpath)
    # if epoch % 50 == 0:
    torch.save(state, savemodelpath + str(epoch) + '_' + str(acc) + '_ckpt.t7')
    pickle.dump(m, open(savemodelpath+'gbtmodel-'+str(fold)+'-'+str(epoch)+'.sav', 'wb'))
    # best_acc = acc
    print('teacc '+str(acc)+' bestacc '+str(best_acc)+' gbttestaccgbt '+str(gbtteacc)+' bestgbt '+str(best_acc_gbt))
    logging.info('teacc '+str(acc)+' bestacc '+str(best_acc)+' ccgbt '+str(gbtteacc)+' bestgbt '+str(best_acc_gbt))

for epoch in range(start_epoch, int(start_epoch+350*neptime)):#200):
    m = train(epoch)
    test(epoch, m)
