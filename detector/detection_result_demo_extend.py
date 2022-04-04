import matplotlib
# %matplotlib inline
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import skimage.measure as measure
import sys
from step1 import *
from full_prep import lumTrans
from layers import nms,iou
resolution = np.array([1, 1, 1])
datapath = '/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-master/training/Data/ForTest/TestSet/'
Preprocesspath = '/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-master/training/Data/ForTest/Preprocess/'
pbbpath = '/home/zhaojie/zhaojie/Lung/code/detector_py3/results/dpn3d26/Test_Prediction_GGO/TestBbox-16-128-160/'
ct_lungsegpath = '/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-master/training/Data/ForTest/TestLungSeg/'
def GetCenterOfNodule(seg_array):
    NoduleBox_list = []
    seg_array = measure.label(seg_array, 4)
    # print(np.max(pred_seg))
    for indexx in range(np.max(seg_array)):
        indexxx = indexx+1
        seg_array_copy = seg_array.copy()
        seg_array_copy[seg_array == indexxx] = 1
        seg_array_copy[seg_array != indexxx] = 0
        z1 = np.any(seg_array_copy, axis=(1, 2))
        ZDstart_slice, ZUend_slice = np.where(z1)[0][[0, -1]] 
        z2 = np.any(seg_array_copy, axis=(0, 1))
        Lstart_slice, Rend_slice = np.where(z2)[0][[0, -1]] 
        z3 = np.any(seg_array_copy, axis=(0, 2))
        Dstart_slice, Uend_slice = np.where(z3)[0][[0, -1]]         
        NoduleBox = [(ZUend_slice + ZDstart_slice)//2,(Uend_slice + Dstart_slice)//2, (Rend_slice + Lstart_slice)//2, max((Uend_slice-Dstart_slice)//2, (Rend_slice-Lstart_slice)//2)]
        # print(ZDstart_slice,ZUend_slice,Dstart_slice,Uend_slice,Lstart_slice,Rend_slice + 1)
        # print('NoduleBox',NoduleBox)
        if NoduleBox[-1] !=0:
            NoduleBox_list.append(NoduleBox)
    return NoduleBox_list
def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord
imglist = sorted([f for f in os.listdir(pbbpath) if f.endswith('_pbb.npy')])
print('imglist',len(imglist))
for i in range(len(imglist)):
    
    Image, Origin, Spacing,isflip = load_itk_image(os.path.join(datapath,imglist[i].split('_')[0] + '.mhd'))
    imgLungmask  = np.load(os.path.join(ct_lungsegpath,imglist[i].split('_')[0] + '_Lungmask.npy'))
    # print(Image.shape,imgLungmask.shape)
    pbb5  = np.load(os.path.join(pbbpath,imglist[i]))
    extendbox = np.load(Preprocesspath + imglist[i].split('_')[0]+'_extendbox.npy', mmap_mode='r')
    save_dir = os.path.join('./GGONodulePrediction',imglist[i].split('_')[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ##########将结节金标准保存
    # noduleMask  = np.load(os.path.join(ct_lungsegpath,imglist[i].split('_')[0] + '_Nodulemask.npy'))[0]
    # NoduleBox_list = GetCenterOfNodule(noduleMask)
    # print('-----------------------------')
    # for seg_num in range(len(NoduleBox_list)):
        # boxS = NoduleBox_list[seg_num]
        # boxXYZ = np.array(boxS[:-1])#去掉直径
        # print('NoduleBox',boxXYZ, extendbox)#
        # boxXYZ = np.array(boxXYZ + np.expand_dims(extendbox[0], 1).T)#对输出加上拓展box的坐标，其实就是恢复为原来的坐标
        # boxXYZ = np.array(boxXYZ * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)#将输出恢复为原来的分辨率，这样就对应了原始数据中的体素坐标
        # pos = VoxelToWorldCoord(boxXYZ, origin, spacing)#将输出转换为世界坐标
        # print('boxXYZ,pos',boxXYZ,pos)
        # ax = plt.subplot(1,1,1)
        # plt.imshow(img[0,boxS[0]],'gray')
        # plt.axis('off')
        # rect = patches.Rectangle((boxS[2]-boxS[3],boxS[1]-boxS[3]),boxS[3]*2,boxS[3]*2,linewidth=2,edgecolor='blue',facecolor='none')
        # ax.add_patch(rect)
        # plt.savefig(os.path.join(save_dir,imglist[i].split('_')[0] + '---' + str(boxS[0]) + '.png'))
        # plt.close()
    if pbb5.shape[0] != 0 :
        max = np.max(pbb5[:,0])
        thes = -10
        # thes = -1
        if max < -1:
            thes = max
        pbb5 = pbb5[pbb5[:,0]>=thes]
        pbb5 = nms(pbb5,0.1)
        # print('pbb5',imglist[i].split('_')[0],pbb5.shape, Image.shape, extendbox.shape)
        pbb = np.array(pbb5[:, :-1])#去掉直径
        print('pbb5', pbb5, Spacing)
        pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:,0], 1).T)#对输出加上拓展box的坐标，其实就是恢复为原来的坐标，我对这个拓展box深恶痛绝
        
        pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(Spacing, 1).T)#将输出恢复为原来的分辨率，这样就对应了原始数据中的体素坐标
        pbb5[:, 2:] = np.array(pbb5[:, 2:] * np.expand_dims(resolution, 1).T / np.expand_dims(Spacing, 1).T)#将输出恢复为原来的分辨率，这样就对应了原始数据中的体素坐标
        pos = VoxelToWorldCoord(pbb[:, 1:], Origin, Spacing)
        
        # print('pos',pos)
        # print('pbb', pbb)
        for box_num in range(pbb.shape[0]):
            DD = pbb5[box_num, -1]*2
            box = pbb[box_num].astype('int')[1:]#取第一个box的xyzd
            # if imgLungmask[0,box[0],box[1],box[2]] != 0:
            if box[0] < Image.shape[1]:
                print('box',box, DD)#array([169, 153,  69,  13])
                
                ax = plt.subplot(1,1,1)
                plt.imshow(Image[box[0]],'gray')
                plt.axis('off')
                rect = patches.Rectangle((box[2]-DD,box[1]-DD),DD*2,DD*2,linewidth=2,edgecolor='red',facecolor='none')
                ax.add_patch(rect)
                plt.savefig(os.path.join(save_dir,imglist[i].split('_')[0] + '---' + str(box[0]) + '.png'))
                if box_num > 0:
                    if box_num < pbb.shape[0]-1:#倒数第二层
                        if box[0] != pbb[box_num + 1].astype('int')[1]:#每个box和下一张不在同一层
                            plt.close()
                    elif box_num == pbb.shape[0]-1:#最后一层
                        plt.close()
                elif box[0] != pbb[box_num + 1].astype('int')[1]:#第一个box和下一张box不在同一层
                    plt.close()