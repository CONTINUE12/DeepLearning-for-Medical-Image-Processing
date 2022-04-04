import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from scipy import misc
class DataBowl3Detector(Dataset):
    def __init__(self, data_dir, split_path, config, phase='train', split_comber=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']       
        self.stride = config['stride']#4
        sizelim = config['sizelim']/config['reso']#2.5/1
        sizelim2 = config['sizelim2']//config['reso']#10/1
        sizelim3 = config['sizelim3']//config['reso']#20/1
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']#{'flip':True,'swap':False,'scale':True,'rotate':False}
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        idcs = split_path # np.load(split_path)
        
        if phase!='test':
            idcs = [f for f in idcs if (f not in self.blacklist)]
        
        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
        # print self.filenames
        self.kagglenames = [f for f in self.filenames]# if len(f.split('/')[-1].split('_')[0])>20]
        # self.lunanames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0])<20]
        # print('-------',idcs)
        labels = []
        
        # print('len(idcs)',len(idcs))
        for idx in idcs:
            # print(data_dir, idx)
            l = np.load(data_dir+idx+'_label.npy', allow_pickle = True)
            # l = np.load(data_dir+idx+'_label.npy')
            if np.all(l==0):#without nodule
                l=np.array([])
            labels.append(l)
        # print('Warning')
        self.sample_bboxes = labels#nodule
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):#[[81.933381 166.26051683000003 43.932055500000004 6.440878725]]
                # print('-------',self.bboxes)
                if len(l) > 0 :
                    for t in l:
                        if t[3]>sizelim:#直径>2.5
                            self.bboxes.append([np.concatenate([[i],t])])
                        if t[3]>sizelim2:#直径>10
                            self.bboxes+=[[np.concatenate([[i],t])]]*2
                        if t[3]>sizelim3:#直径>20
                            self.bboxes+=[[np.concatenate([[i],t])]]*4
            self.bboxes = np.concatenate(self.bboxes,axis = 0)# many boxes
        # print('-------',self.bboxes, self.sample_bboxes)
        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

    def __getitem__(self, idx,split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time
        isRandomImg  = False
        if self.phase !='test':
            # print('idx, len(self.bboxes)',idx, len(self.bboxes))
            if idx>=len(self.bboxes):
                isRandom = True
                idx = idx%len(self.bboxes)
                isRandomImg = np.random.randint(2)#随机产生0和1
            else:
                isRandom = False
        else:
            isRandom = False
        # print('Warning1')
        if self.phase != 'test':
            if not isRandomImg:
                bbox = self.bboxes[idx]#num,x,y,z,r,第一个参数代表的是第几个CT序列（trainlist排列好的）后面代表的是xyz和直径。当直径大于2.5写1行，大于10写1+2行，大于20写1+2+4行
                filename = self.filenames[int(bbox[0])]#第int(bbox[0])序列的路径（包含名称_clearn.npy）
                imgs = np.load(filename)
                
                bboxes = self.sample_bboxes[int(bbox[0])]#self.sample_bboxes就是800CT对应的结节的xyz和直径，是一个列表list（可能是空，也可能是多行）
                isScale = self.augtype['scale'] and (self.phase=='train')#True
                # print('---bbox---',bbox)
                # print('---bboxes---',isRandom)
                #imgs是_clearn.npy导入的3d矩阵(仅仅包含肺部并且分辨率为1×1*1mm)/bbox代表[556 74.7895776 249.75681577 267.72357450000004 10.57235285],bbox[1:]代表[74.7895776 249.75681577 267.72357450000004 10.57235285]
                # bboxes代表[[74.7895776 249.75681577 267.72357450000004 10.57235285],[165.8177988 172.30634478 79.42561491000001 20.484109399999998]]等于或包含bbox[1:].
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes,isScale,isRandom)
                if self.phase=='train' and not isRandom:
                     sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                        ifflip = self.augtype['flip'], ifrotate=self.augtype['rotate'], ifswap = self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.kagglenames))
                filename = self.kagglenames[randimid]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes,isScale=False,isRand=True)
            # print('Warning2')
            label = self.label_mapping(sample.shape[1:], target, bboxes, filename)
            sample = (sample.astype(np.float32)-128)/128
            
            # print('label',np.unique(label),label.shape,label.dtype)#array([-1.,  0.], dtype=float32), (24, 24, 24, 3, 5), dtype('float32')
            # print('coord',coord.shape, coord.dtype)#(3, 24, 24, 24)
            #if filename in self.kagglenames and self.phase=='train':
            #    label[label==-1]=0
            return torch.from_numpy(sample), torch.from_numpy(label), coord
        else:#test
            imgs = np.load(self.filenames[idx])
            # print('imgs.shape',imgs.shape)
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]#(197, 181, 224)
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0,0],[0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',constant_values = self.pad_value)#能被stride整除
            # print('1------pad',self.stride, pz,ph,pw)
            xx,yy,zz = np.meshgrid(np.linspace(-0.5,0.5,imgs.shape[1]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[2]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[3]/self.stride),indexing ='ij')
            coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')
            # print('------1',imgs.shape)
            imgs, nzhw = self.split_comber.split(imgs)
            # print('------2',imgs.shape)
            # print('------',self.split_comber.max_stride,self.stride)
            
            coord2, nzhw2 = self.split_comber.split(coord,
                                                   side_len = self.split_comber.side_len//self.stride,
                                                   max_stride = self.split_comber.max_stride//self.stride,
                                                   margin = self.split_comber.margin//self.stride)
            assert np.all(nzhw==nzhw2)
            imgs = (imgs.astype(np.float32)-128)/128
            # print('sample.shape',imgs.shape,coord2.shape)
            return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nzhw)

    def __len__(self):
        # print('Warning00')
        if self.phase == 'train':
            # print(type(len(self.bboxes)),type(1-self.r_rand))
            # print(len(self.bboxes)/(1-self.r_rand))
            # print(len(self.bboxes),(1-self.r_rand))
            # print('Warning11')
            # return len(self.bboxes)/(1-self.r_rand)
            return int(len(self.bboxes)/(1-self.r_rand))
        elif self.phase =='val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)

def augment(sample, target, bboxes, coord, ifflip = True, ifrotate=True, ifswap = True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                coord = rotate(coord,angle1,axes=(2,3),reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat,box[1:3]-size/2)+size/2
            else:
                counter += 1
                if counter ==3:
                    break
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]
            
    if ifflip:
#         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
        for ax in range(3):
            if flipid[ax]==-1:
                target[ax] = np.array(sample.shape[ax+1])-target[ax]
                bboxes[:,ax]= np.array(sample.shape[ax+1])-bboxes[:,ax]
    return sample, target, bboxes, coord 

class Crop(object):#imgs, bbox[1:], bboxes,isScale,isRandom
    def __init__(self, config):
        self.crop_size = config['crop_size']#[96, 96, 96]
        self.bound_size = config['bound_size']#12
        self.stride = config['stride']#4
        self.pad_value = config['pad_value']#170
    def __call__(self, imgs, target, bboxes,isScale=False,isRand=False):
        if isScale:
            radiusLim = [8.,120.]
            scaleLim = [0.75,1.25]
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1]) ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])]#target代表bbox[1:]代表[74.7895776 249.75681577 267.72357450000004 10.57235285]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]#np.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')#根据实际结节直径大小调整crop_size大小,target[3]小crop_size变大.target[3]大crop_size变小
        else:
            crop_size=self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)#目标结节
        bboxes = np.copy(bboxes)#这个ct含有的所有结节
        # print('---crop---',target)
        start = []
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r)+ 1 - bound_size
                e = np.ceil (target[i] + r)+ 1 + bound_size - crop_size[i] 
            else:#
                s = np.max([imgs.shape[i+1]-crop_size[i]//2,imgs.shape[i+1]//2+bound_size])
                e = np.min([crop_size[i]//2, imgs.shape[i+1]//2-bound_size])
                target = np.array([np.nan,np.nan,np.nan,np.nan])
                # print(s,e)
            if s>e:
                start.append(np.random.randint(e,s))#!
            else:
                start.append(int(target[i])-crop_size[i]//2+np.random.randint(-bound_size//2,bound_size//2))#求取结节的3d立方矩阵最靠近原点的点
                
                
        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5#将normstart放到3d块[-0.5:0.5,-0.5:0.5,-0.5:0.5]对应实际[-imgs.shape[1]/2:imgs.shape[1]/2,-imgs.shape[2]/2:imgs.shape[2]/2,-imgs.shape[1]/2:imgs.shape[1]/2,-imgs.shape[3]/2:imgs.shape[3]/2]
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],self.crop_size[0]//self.stride),
                           np.linspace(normstart[1],normstart[1]+normsize[1],self.crop_size[1]//self.stride),#np.linspace创建等差数列
                           np.linspace(normstart[2],normstart[2]+normsize[2],self.crop_size[2]//self.stride),indexing ='ij')#可以这么理解，meshgrid函数用两个坐标轴上的点在平面上画网格(3d也是如此)
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')
        
        pad = []
        pad.append([0,0])
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,#以结节位置为中心来截取
            max(start[0],0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1],imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2],imgs.shape[3])]
        # print('---crop0---',crop.shape)
        crop = np.pad(crop,pad,'constant',constant_values =self.pad_value)#越界进行填充
        # print('---crop---',max(start[0],0),min(start[0] + crop_size[0],imgs.shape[1]))
        # print('---crop1---',crop.shape)
        # print(stop)
        for i in range(3):
            target[i] = target[i] - start[i]#目标结节位置已经减去目标结节3d立方矩阵最靠近原点的开始点
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]#同一ct的所有结节位置已经减去该目标结节3d立方矩阵最靠近原点的开始点
                
        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop,[1,scale,scale,scale],order=1)
            newpad = self.crop_size[0]-crop.shape[1:][0]
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
            elif newpad>0:
                pad2 = [[0,0],[0,newpad],[0,newpad],[0,newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i]*scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j]*scale
        # for i in range(crop.shape[1]):
            # name = str(i) + '.png'
            # misc.imsave(os.path.join('../Visualize/crop/', name), crop[0][i])
        # print('target',target)
        # print('bboxes',bboxes)
        return crop, target, bboxes, coord#1*96*96*96/可能是[nan,nan,nan,nan]/可能是空/3*24*24*24
    
class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])#4
        self.num_neg = int(config['num_neg'])#800
        self.th_neg = config['th_neg']#0.02
        self.anchors = np.asarray(config['anchors'])#5.,10.,20.
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train']#0.5
        elif phase == 'val':
            self.th_pos = config['th_pos_val']

            
    def __call__(self, input_size, target, bboxes, filename):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos
        
        output_size = []
        for i in range(3):
            if input_size[i] % stride != 0:
                print('filename',filename)
            # assert(input_size[i] % stride == 0) 
            output_size.append(input_size[i] // stride)
        
        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)
        offset = ((stride.astype('float')) - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        # print('---',oz)
        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                # print()
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)#选择iou大于0.02的框
                label[iz, ih, iw, i, 0] = 0#把相关类别标签标签你从-1改为0（最后那维0是类别)最后的3*5相当于（当i=0时,  [[ 0. -1. -1. -1. -1.],[-1. -1. -1. -1. -1.],[-1. -1. -1. -1. -1.]])可以理解为24*24*24个3*5的矩阵罗列起来，不同的只是外面包的中括号层数的不同

        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1#产生800个负标签

        if np.isnan(target[0]):
            return label
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)#选择iou大于0.5的框
            iz.append(iiz)#可能是[[1, 2, 3], [2, 3, 5, 6]]
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)#最后[1 2 3 2 3 5 6]
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True 
        if len(iz) == 0:
            pos = []
            for i in range(3):
                # print('0',target[i])
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(iz)), 1)[0]#从中随机的选取一个
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        # print('0',target[3]/anchors[pos[3]])
        dd = np.log(target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]#产生一个正标签
        return label        

def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, d = bbox
    max_overlap = min(d, anchor)
    
    # print(np.power(max(d, anchor), 3) * th,max_overlap,max_overlap)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    # print('bbox',bbox, min_overlap, max_overlap)
    # print(bb)
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]
        
        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]
            
        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
        
        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis = 1)
        
        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0
        
        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))
        
        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))
        # print('overlap',e0,e1,s0,s1)
        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        # print(intersection, union)
        iou = intersection / union

        mask = iou >= th
        #if th > 0.4:
         #   if np.sum(mask) == 0:
          #      print(['iou not large', iou.max()])
           # else:
            #    print(['iou large', iou[mask]])
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw

def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

