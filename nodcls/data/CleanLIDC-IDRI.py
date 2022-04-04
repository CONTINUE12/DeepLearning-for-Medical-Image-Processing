import os
from os.path import join, getsize
import numpy as np

def getdirsize(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([getsize(join(root, name)) for name in files])
    return size

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        print('c_path',c_path)
        if os.path.isdir(c_path):
            if len(os.listdir(c_path)) == 0:
                os.removedirs(c_path)
            else:
                del_file(c_path)
            
        else:
            # print(c_path)
            os.remove(c_path)
            # print(c_path.split('/')[:-1])
            # os.removedirs(c_path.split('/')[:-1])
    # os.removedirs(path) 
if __name__ == '__main__':
    '''#删除多余的文件
    MainPath = '/home/zhaojie/zhaojie/Lung/data/LIDC-IDRI/'
    for pid0 in sorted(os.listdir(MainPath)):
        MovePath = []
        sizeList = []
        if len(os.listdir(MainPath + pid0)) !=1 :
            print('0',MainPath + pid0)
            # print('1',os.listdir(MainPath + pid0))
            for pid1 in sorted(os.listdir(MainPath + pid0)): 
                MovePath.append(MainPath + pid0 + '/' + pid1)
                
                sizeList.append(getdirsize(MainPath + pid0 + '/' + pid1))
            # print('There are %.3f' % (getdirsize(MovePath[0]) / 1024 / 1024), MovePath[0])
            # print('There are %.3f' % (getdirsize(MovePath[1]) / 1024 / 1024), MovePath[1])
			###################################
            # print('!!!!!removed',MovePath[np.argsort(sizeList)[0]].split('/')[-1])
            # del_file(MovePath[np.argsort(sizeList)[0]])
            # for pid2 in os.listdir(MovePath[np.argsort(sizeList)[0]]):
                # os.removedirs(MovePath[np.argsort(sizeList)[0]] + '/' + pid2)
            # print(stop) 
            ######################################
                if len(os.listdir(MainPath + pid0+ '/' +pid1)) !=1 :
                    # print('2',os.listdir(MainPath + pid0 +'/' + pid1))
                    
                    for pid2 in sorted(os.listdir(MainPath+pid0 + '/'+pid1)):
                        if len(os.listdir(MainPath + pid0+ '/' +pid1)) !=1 :
                            print('3',os.listdir(MainPath + pid0 +'/' + pid1 + '/' + pid2))
                            print(stop)
    '''
    '''#修改LIDC的文件夹名称
    rootDir = '/home/zhaojie/zhaojie/Lung/data/LIDC-IDRI/'
    mapfname = 'LIDC-IDRI-mappingLUNA16'
    sidmap = {}
    fid = open(mapfname, 'r')
    line = fid.readline()
    line = fid.readline()
    while line:
        pidlist = line.split(' ')
        print('pidlist',pidlist)#['LIDC-IDRI-1011', '1.3.6.1.4.1.14519.5.2.1.6279.6001.287560874054243719452635194040',倒数第二 '1.3.6.1.4.1.14519.5.2.1.6279.6001.272123398257168239653655006815', '\n']最后
        pid = pidlist[0] 
        stdid = pidlist[1] 
        srsid = pidlist[2]
        # if pid == 'LIDC-IDRI-0332':#有2个不同的CT，注意保留
        for pid0 in os.listdir(os.path.join(rootDir, pid)):
            oldname = os.path.join(rootDir, pid, pid0)
            newname = os.path.join(rootDir, pid, stdid)
            print('oldname, newname',oldname, newname)
            os.rename(oldname, newname)
            for pid1 in os.listdir(os.path.join(rootDir, pid, stdid)):
                oldname1 = os.path.join(rootDir, pid, stdid, pid1)
                newname1 = os.path.join(rootDir, pid, stdid, srsid)
                print('oldname1, newname1', newname1)
                os.rename(oldname1, newname1)
                print(stop)
        line = fid.readline()
    fid.close()
    '''
    
    import shutil
    preprocesspath = '/home/zhaojie/zhaojie/Lung/data/luna16/LUNA16PROPOCESSPATH/'
    savepath = '/home/zhaojie/zhaojie/Lung/data/luna16/cls/test0-5/'
    # savepath = '/home/zhaojie/zhaojie/Lung/data/luna16/cls/train6-9/'
    for setidx in range(6):
    # for setidx in range(6,10):
       print('process subset', setidx)
       filelist = [f for f in os.listdir(os.path.join(preprocesspath + 'subset' + str(setidx))) if f.endswith('clean.npy')]
       print('filelist',len(filelist))
       for i in range(len(filelist)):
           shutil.copy(os.path.join(preprocesspath + 'subset' + str(setidx),filelist[i]),savepath)
