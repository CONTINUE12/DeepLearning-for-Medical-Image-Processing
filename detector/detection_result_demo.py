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
datapath = '/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-master/training/Data/ForTest/Preprocess/'
segpath = '/home/zhaojie/zhaojie/Lung/DSB_Code/DSB2017-master/training/Data/ForTest/TestSeg/'
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

imglist = sorted([f for f in os.listdir(pbbpath) if f.endswith('_pbb.npy')])
print('imglist',len(imglist))
for i in range(len(imglist)):
    img  = np.load(os.path.join(datapath,imglist[i].split('_')[0] + '_clean.npy'))
    imgLungmask  = np.load(os.path.join(ct_lungsegpath,imglist[i].split('_')[0] + '_Lungmask.npy'))
    pbb  = np.load(os.path.join(pbbpath,imglist[i]))
    save_dir = os.path.join('./GGONodulePrediction',imglist[i].split('_')[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ##########将结节金标准保存
    # noduleMask  = np.load(os.path.join(ct_lungsegpath,imglist[i].split('_')[0] + '_Nodulemask.npy'))[0]
    # NoduleBox_list = GetCenterOfNodule(noduleMask)
    print('-----------------------------')
    # for seg_num in range(len(NoduleBox_list)):
        # boxS = NoduleBox_list[seg_num]
        # print('NoduleBox',boxS)#
        # ax = plt.subplot(1,1,1)
        # plt.imshow(img[0,boxS[0]],'gray')
        # plt.axis('off')
        # rect = patches.Rectangle((boxS[2]-boxS[3],boxS[1]-boxS[3]),boxS[3]*2,boxS[3]*2,linewidth=2,edgecolor='blue',facecolor='none')
        # ax.add_patch(rect)
        # plt.savefig(os.path.join(save_dir,imglist[i].split('_')[0] + '---' + str(boxS[0]) + '.png'))
        # plt.close()
    if pbb.shape[0] != 0 :
        max = np.max(pbb[:,0])
        thes = -10
        # thes = -1
        if max < -1:
            thes = max
        pbb = pbb[pbb[:,0]>=thes]
        pbb = nms(pbb,0.1)
        print('pbb',imglist[i].split('_')[0],pbb.shape, img.shape)
        
        for box_num in range(pbb.shape[0]):
            box = pbb[box_num].astype('int')[1:]#取第一个box的xyzd
            # if imgLungmask[0,box[0],box[1],box[2]] != 0:
            if box[0] < img.shape[1]:
                print('box',box)#array([169, 153,  69,  13])
                
                ax = plt.subplot(1,1,1)
                plt.imshow(img[0,box[0]],'gray')
                plt.axis('off')
                rect = patches.Rectangle((box[2]-box[3],box[1]-box[3]),box[3]*2,box[3]*2,linewidth=2,edgecolor='red',facecolor='none')
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