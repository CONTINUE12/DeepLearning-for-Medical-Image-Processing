import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
# from noduleCADEvaluationLUNA16 import noduleCADEvaluation
from noduleCADEvaluationLUNA16compare import noduleCADEvaluation
import os 
import csv 
from multiprocessing import Pool
import functools
import SimpleITK as sitk
fold = 4
trainnum = 5
annotations_filename = '/home/zhaojie/zhaojie/Lung/code/evaluationScript/10FoldCsvFiles/annotations' +str(fold) + '.csv'# path for ground truth annotations for the fold
annotations_excluded_filename = '/home/zhaojie/zhaojie/Lung/code/evaluationScript/10FoldCsvFiles/annotations_excluded' +str(fold) + '.csv'# path for excluded annotations for the fold
seriesuids_filename = '/home/zhaojie/zhaojie/Lung/code/evaluationScript/10FoldCsvFiles/seriesuids' +str(fold) + '.csv'# path for seriesuid for the fold
results_path = '/home/zhaojie/zhaojie/Lung/code/detector_py3/results/dpn3d26/retrft96' + str(trainnum) + '/val'#val' #val' ft96'+'/val'#
sideinfopath = '/home/zhaojie/zhaojie/Lung/data/luna16/LUNA16PROPOCESSPATH/subset'+str(fold)+'/'
datapath = '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset'+str(fold)+'/'
maxeps = 995 #03 #150 #100#100

# eps = range(1, maxeps+1, 1)#6,7,1)#5,151,5)#5,151,5)#76,77,1)#40,41,1)#76,77,1)#1,101,1)#17,18,1)#38,39,1)#1, maxeps+1, 1) #maxeps+1, 1)
eps = range(995, maxeps+1, 1)#6,7,1)#5,151,5)#5,151,5)#76,77,1)#40,41,1)#76,77,1)#1,101,1)#17,18,1)#38,39,1)#1, maxeps+1, 1) #maxeps+1, 1)
detp = [-1.5, -1]#, -0.5, 0]#, 0.5, 1]#, 0.5, 1] #range(-1, 0, 1)
# detp = [-1]#, -0.5, 0]#, 0.5, 1]#, 0.5, 1] #range(-1, 0, 1)
isvis = False #True
nmsthresh = 0.1
nprocess = 38#4
use_softnms = False
frocarr = np.zeros((maxeps, len(detp)))
firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']

def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord
def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip
def iou(box0, box1):
    r0 = box0[3] / 2#半径
    s0 = box0[:3] - r0#xyz的左上角
    e0 = box0[:3] + r0#xyz的右上角
    r1 = box1[3] / 2
    s1 = box1[:3] - r1#xyz的左上角
    e1 = box1[:3] + r1#xyz的右上角
    overlap = []
    for i in range(len(s0)):#3个
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))#最小的右下角-最大的左下角=((x1-x0),(y1-y0),(z1-z0))
    intersection = overlap[0] * overlap[1] * overlap[2]#(x1-x0)*(y1-y0)*(z1-z0)
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection#体积就是3个直径的乘积
    return intersection / union
def nms(output, nms_th):
    if len(output) == 0:
        return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def convertcsv(bboxfname, bboxpath, detp):#给定pbb.npy的文件名，pbb.npy的路径，阈值
    # print(datapath, sideinfopath, bboxpath)#加载原始数据
    #/public/share/jiezhao/Minerva/Lung/data/luna16/subset_data/subset9/ /public/share/jiezhao/Minerva/Lung/data/luna16/LUNA16PROPOCESSPATH/subset9/ /public/share/jiezhao/Minerva/Lung/DeepLung-master/detector_py3/results/dpn3d26/retrft960/val1/
    sliceim,origin,spacing,isflip = load_itk_image(datapath+bboxfname[:-8]+'.mhd')#加载原始数据
    origin = np.load(sideinfopath+bboxfname[:-8]+'_origin.npy', mmap_mode='r')#以下几行加载预处理后的坐标原点，分辨率，拓展box
    spacing = np.load(sideinfopath+bboxfname[:-8]+'_spacing.npy', mmap_mode='r')
    resolution = np.array([1, 1, 1])
    extendbox = np.load(sideinfopath+bboxfname[:-8]+'_extendbox.npy', mmap_mode='r')
    
    # if str(bboxfname) == '1.3.6.1.4.1.14519.5.2.1.6279.6001.265960756233787099041040311282_pbb.npy':
        print(bboxpath+bboxfname)
        #加载pbb.npy文件
        pbb = np.load(bboxpath+bboxfname, mmap_mode='r')#加载pbb.npy文件
        print('pbb.shape',pbb.shape)#(267, 5)
        pbbold = np.array(pbb[pbb[:,0] > detp])#根据阈值过滤掉概率低的
        pbbold = np.array(pbbold[pbbold[:,-1] > 3])  # add new 9 15#根据半径过滤掉小于3mm的
        pbbold = pbbold[np.argsort(-pbbold[:,0])][:1000] #这条是我加上的，取概率值前1000的结节作为输出，不然直接进行nms耗时太长
        # print('pbbold.shape',pbbold.shape)
        # pbb = np.array(pbb[:K, :4])
        # print pbbold.shape1
        # if use_softnms:
        #     keep = cpu_soft_nms(pbbold, method=2) # 1 for linear weighting, 2 for gaussian weighting
        #     pbb = np.array(pbbold[keep]) #cpu_soft_nms(pbbold)
        # else:
        pbb = nms(pbbold, nmsthresh)#对输出的结节进行nms
        print('len(pbb), pbb[0]',pbb.shape)
        # print bboxfname, pbbold.shape, pbb.shape, pbbold.shape
        pbb = np.array(pbb[:, :-1])#去掉直径
        # print(stop)
        pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:,0], 1).T)#对输出加上拓展box的坐标，其实就是恢复为原来的坐标，我对这个拓展box深恶痛绝
        pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)#将输出恢复为原来的分辨率，这样就对应了原始数据中的体素坐标
        if isflip:#如果有翻转的情况，将坐标翻转（我理解是这样的，原始数据有翻转的情况，但是label还是未翻转的label，那么将label翻转，所以模型的输出也是翻转的，现在要再翻转回去，与label对应）
            Mask = np.load(sideinfopath+bboxfname[:-8]+'_mask.npy', mmap_mode='r')
            pbb[:, 2] = Mask.shape[1] - pbb[:, 2]
            pbb[:, 3] = Mask.shape[2] - pbb[:, 3]
        pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)#将输出转换为世界坐标
        rowlist = []
        # print pos.shape
        for nk in range(pos.shape[0]): # pos[nk, 2], pos[nk, 1], pos[nk, 0]
            rowlist.append([bboxfname[:-8], pos[nk, 2], pos[nk, 1], pos[nk, 0], 1/(1+np.exp(-pbb[nk,0]))])#现在依次将文件名，z,y,x，概率（经过sigmoid处理）写入rowlist，每行都是一个输出结节
        # print ('convertcsv-len(rowlist), len(rowlist[0])',len(rowlist), len(rowlist[0]))
        return rowlist#bboxfname[:-8], pos[:K, 2], pos[:K, 1], pos[:K, 0], 1/(1+np.exp(-pbb[:K,0]))
def getfrocvalue(results_filename):
    return noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,'./outputDir/')#vis=False)
    # return noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,'./', vis=isvis)#vis=False)
p = Pool(nprocess)
#结果就是每一个epoch都生成一个csv文件，存放80多个测试病例的预测结节位置及概率
def getcsv(detp, eps):#给定阈值和epoch
    for ep in eps:#针对每个epoch
        bboxpath = results_path + str(ep) + '/'#找到每个epoch的路径
        for detpthresh in detp:
            print('ep', ep, 'detp', detpthresh, bboxpath)
            f = open(bboxpath + 'predanno'+ str(detpthresh) + '.csv', 'w')#根据阈值分别创建与之对应的文件
            fwriter = csv.writer(f)
            fwriter.writerow(firstline)#写入第一行，包括用户id,结节坐标x,y,z,结节概率p
            fnamelist = []
            for fname in os.listdir(bboxpath):
                if fname.endswith('_pbb.npy'):#找到以_pbb.npy结尾的文件（输出的结节预测值），添加进文件列表
                    fnamelist.append(fname)
                    # print fname
                    # for row in convertcsv(fname, bboxpath, k):
                        # fwriter.writerow(row)
            # # return
            print('len(fnamelist)',len(fnamelist))
            
            predannolist = p.map(functools.partial(convertcsv, bboxpath=bboxpath, detp=detpthresh), fnamelist)#这个函数对convertcsv函数进行修饰，其实就是预设定几个参数，不用再输入
            # print len(predannolist), len(predannolist[0])
            
            for predanno in predannolist:
                # print predanno
                for row in predanno:
                    # print row
                    fwriter.writerow(row)
            f.close()
getcsv(detp, eps)
print(stop)
def getfroc(detp, eps):
    maxfroc = 0
    maxep = 0
    for ep in eps:#对每个epoch分别处理
        bboxpath = results_path + str(ep) + '/'
        predannofnamalist = []
        # print('detp, bboxpath',detp, bboxpath)#[-1.5, -1]  /public/share/jiezhao/Minerva/Lung/DeepLung-master/detector_py3/results/dpn3d26/retrft960/val1/
        #此处的detp就是阈值，只不过这里采用的是一个阈值列表，就我自己而言，我采用的阈值是-0.125，列表中只有一个元素
        for detpthresh in detp:
            predannofnamalist.append(bboxpath + 'predanno'+ str(detpthresh) + '.csv')
            print('DONE!',detpthresh)
        # print('predannofnamalist',predannofnamalist)#['/public/share/jiezhao/Minerva/Lung/DeepLung-master/detector_py3/results/dpn3d26/retrft960/val199/predanno-1.5.csv', '/public/share/jiezhao/Minerva/Lung/DeepLung-master/detector_py3/results/dpn3d26/retrft960/val199/predanno-1.csv']

        
        froclist = p.map(getfrocvalue, predannofnamalist)#调用getfrocvalue求取froc值
        # print('maxfroc0', ep, max(froclist))
        # print('max(froclist)',froclist)
        if maxfroc < max(froclist):
            maxep = ep
            maxfroc = max(froclist)
        # print('maxfroc1',maxfroc)
        for detpthresh in detp:
            # print((ep-eps[0])//(eps[1]-eps[0]), int((detpthresh-detp[0])/(detp[1]-detp[0])))
            # print((ep-eps[0])/(eps[1]-eps[0]), int((detpthresh-detp[0])/(detp[1]-detp[0])))
            frocarr[(ep-eps[0])//(eps[1]-eps[0]), int((detpthresh-detp[0])/(detp[1]-detp[0]))] = froclist[int((detpthresh-detp[0])/(detp[1]-detp[0]))]
            # print('ep', ep, 'detp', detpthresh, froclist[int((detpthresh-detp[0])/(detp[1]-detp[0]))])
    print('maxfroc, maxep', maxfroc, maxep)
getfroc(detp, eps)
p.close()
fig = plt.imshow(frocarr.T)
plt.colorbar()
plt.xlabel('# Epochs')
plt.ylabel('# Detection Prob.')
xtick = detp #[36, 37, 38, 39, 40]
plt.yticks(range(len(xtick)), xtick)
ytick = eps #range(51, maxeps+1, 2)
plt.xticks(range(len(ytick)), ytick)
plt.title('Average FROC')
plt.savefig(results_path+'frocavg.png')
np.save(results_path+'frocavg.npy', frocarr)
frocarr = np.load(results_path+'frocavg.npy', 'r')
froc, x, y = 0, 0, 0
for i in range(frocarr.shape[0]):
    for j in range(frocarr.shape[1]):
        if froc < frocarr[i,j]:
            froc, x, y = frocarr[i,j], i, j
print('FINISH:',fold, froc, x, y)
