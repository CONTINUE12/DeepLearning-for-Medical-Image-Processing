import pandas as pd 
import SimpleITK as sitk
import os
import os.path
import numpy as np
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
def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord
# read map file
mapfname = 'LIDC-IDRI-mappingLUNA16'
sidmap = {}
fid = open(mapfname, 'r')
line = fid.readline()
line = fid.readline()
while line:
	pidlist = line.split(' ')
	# print('pidlist',pidlist[0],pidlist[2])#['LIDC-IDRI-1011', '1.3.6.1.4.1.14519.5.2.1.6279.6001.287560874054243719452635194040',倒数第二 '1.3.6.1.4.1.14519.5.2.1.6279.6001.272123398257168239653655006815', '\n']最后
	pid = pidlist[0] 
	stdid = pidlist[1] 
	srsid = pidlist[2]
	# if pid == 'LIDC-IDRI-0332':
	    # print('pidlist',pidlist)
	if srsid not in sidmap:
		sidmap[srsid] = [pid, stdid]
	else:
		assert sidmap[srsid][0] == pid
		assert sidmap[srsid][1] == stdid
	line = fid.readline()
fid.close()
# print(sidmap['LIDC-IDRI-0332'])
# print('sidmap',sidmap)#'1.3.6.1.4.1.14519.5.2.1.6279.6001.153646219551578201092527860224'(luna16): ['LIDC-IDRI-1012'(LIDC-IDRI), '1.3.6.1.4.1.14519.5.2.1.6279.6001.676549258486738448212921834668']

# read luna16 annotation
colname = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']
lunaantframe = pd.read_csv('../annotations/annotations.csv', names=colname)
srslist = lunaantframe.seriesuid.tolist()[1:]#1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860
cdxlist = lunaantframe.coordX.tolist()[1:]
cdylist = lunaantframe.coordY.tolist()[1:]
cdzlist = lunaantframe.coordZ.tolist()[1:]
dimlist = lunaantframe.diameter_mm.tolist()[1:]

lunaantdict = {}#'1.3.6.1.4.1.14519.5.2.1.6279.6001.970264865033574190975654369557': [[-109.4285094, 59.59670322, -162.9329603, 3.841244999]]
for idx in range(len(srslist)):
	vlu = [float(cdxlist[idx]), float(cdylist[idx]), float(cdzlist[idx]), float(dimlist[idx])]
	if srslist[idx] in lunaantdict:
		lunaantdict[srslist[idx]].append(vlu)
	else:
		lunaantdict[srslist[idx]] = [vlu]

# convert luna16 annotation to LIDC-IDRI annotation space
from multiprocessing import Pool
lunantdictlidc = {}
for fold in range(10):
	mhdpath = '/home/zhaojie/zhaojie/Lung/data/luna16/subset_data/subset'+str(fold)
	print('fold', fold)
	def getvoxcrd(fname):
		sliceim,origin,spacing,isflip = load_itk_image(os.path.join(mhdpath, fname))
		lunantdictlidc[fname[:-4]] = []
		voxcrdlist = []
		for lunaant in lunaantdict[fname[:-4]]:
			voxcrd = worldToVoxelCoord(lunaant[:3][::-1], origin, spacing)
			
			voxcrd[0] = sliceim.shape[0] - voxcrd[0]
			voxcrdlist.append(voxcrd)
		return voxcrdlist
	p = Pool(30)
	fnamelist = []
	for fname in os.listdir(mhdpath):
		if fname.endswith('.mhd') and fname[:-4] in lunaantdict:
			fnamelist.append(fname)
	voxcrdlist = p.map(getvoxcrd, fnamelist)
	listidx = 0
	for fname in os.listdir(mhdpath):
		if fname.endswith('.mhd') and fname[:-4] in lunaantdict:
			lunantdictlidc[fname[:-4]] = []
			for subidx, lunaant in enumerate(lunaantdict[fname[:-4]]):
				# voxcrd = worldToVoxelCoord(lunaant[:3][::-1], origin, spacing)
				# voxcrd[-1] = sliceim.shape[0] - voxcrd[0]
				lunantdictlidc[fname[:-4]].append([lunaant, voxcrdlist[listidx][subidx]])
			listidx += 1
	p.close()
np.save('lunaantdictlidc.npy', lunantdictlidc)#'1.3.6.1.4.1.14519.5.2.1.6279.6001.220596530836092324070084384692': [[[-84.393952, 246.9323285, -508.1886279, 23.57156231], array([244.9113627 , 354.70570816, 126.0886373 ])]],

# read LIDC dataset
lunantdictlidc = np.load('lunaantdictlidc.npy', allow_pickle=True).item()
import xlrd
lidccsvfname = '../annotations/list3.2.xls'
antdict = {}
wb = xlrd.open_workbook(os.path.join(lidccsvfname))
# wb = pd.read_csv(os.path.join(lidccsvfname))
for s in wb.sheets():
	if s.name == 'list3.2':
		for row in range(1, s.nrows):#2636
			# print('row',s.cell(row, 2).value)#roi	volume	eq. diam.	x loc.	y loc.	slice no.1	6459.75	23.107	317	367	43

			# print(stop)
			valuelist = [int(s.cell(row, 2).value), s.cell(row, 3).value, s.cell(row, 4).value, \
			    int(s.cell(row, 5).value), int(s.cell(row, 6).value), int(s.cell(row, 7).value)]
			assert abs(s.cell(row, 1).value - int(s.cell(row, 1).value)) < 1e-8
			assert abs(s.cell(row, 2).value - int(s.cell(row, 2).value)) < 1e-8
			assert abs(s.cell(row, 5).value - int(s.cell(row, 5).value)) < 1e-8
			assert abs(s.cell(row, 6).value - int(s.cell(row, 6).value)) < 1e-8 
			assert abs(s.cell(row, 7).value - int(s.cell(row, 7).value)) < 1e-8
			for col in range(9, 16):
				if s.cell(row, col).value != '':
					if isinstance(s.cell(row, col).value, float):
						valuelist.append(int(s.cell(row, col).value))
						assert abs(s.cell(row, col).value - int(s.cell(row, col).value)) < 1e-8
					else:
						valuelist.append(s.cell(row, col).value)
			# print('valuelist',valuelist)# [1, 6459.75, 23.107, 317, 367, 43, 'IL057_127364', 'Nodule 001', 'MI014_12127', 0]
			if s.cell(row, 0).value+'_'+str(int(s.cell(row, 1).value)) not in antdict:
				antdict[s.cell(row, 0).value+'_'+str(int(s.cell(row, 1).value))] = [valuelist]
			else:
				antdict[s.cell(row, 0).value+'_'+str(int(s.cell(row, 1).value))].append(valuelist)
			# print('antdict',antdict)#{'0001_3000566': [[1, 6459.75, 23.107, 317, 367, 43, 'IL057_127364', 'Nodule 001', 'MI014_12127', 0]]}
			
# update LIDC annotation with series number, rather than scan id
import pydicom
LIDCpath = '/home/zhaojie/zhaojie/Lung/data/LIDC-IDRI/'
antdictscan = {}
for k, v in antdict.items():
	pid, scan = k.split('_')
	# print('pid, scan',pid, scan)#0001 3000566
	hasscan = False
	for sdu in os.listdir(os.path.join(LIDCpath, 'LIDC-IDRI-'+pid)):
		# print(os.path.join(*[LIDCpath, 'LIDC-IDRI-'+pid, sdu]))#/home/zhaojie/zhaojie/Lung/data/LIDC-IDRI/LIDC-IDRI-1012/1.3.6.1.4.1.14519.5.2.1.6279.6001.676549258486738448212921834668
		for srs in os.listdir(os.path.join(*[LIDCpath, 'LIDC-IDRI-'+pid, sdu])):
			# print('srs',srs)#1.3.6.1.4.1.14519.5.2.1.6279.6001.153646219551578201092527860224
			if srs.endswith('.npy'):
				print('npy', pid, scan, srs)
				continue
			# print(os.path.join(*[LIDCpath, 'LIDC-IDRI-'+pid, sdu, srs, '000006.dcm']))
			RefDs = pydicom.read_file(os.path.join(*[LIDCpath, 'LIDC-IDRI-'+pid, sdu, srs, '000006.dcm']))
			# print('str(RefDs[0x20, 0x11].value)',scan, str(RefDs[0x20, 0x11].value)) 
			# print(stop)
			if str(RefDs[0x20, 0x11].value) == scan or scan == '0': 
				if hasscan: print('rep', pid, sdu, srs)
				hasscan = True
				# print('pid+srs', pid+'_'+srs)#1012_1.3.6.1.4.1.14519.5.2.1.6279.6001.153646219551578201092527860224
				antdictscan[pid+'_'+srs] = v
				break
	if not hasscan: print('not found', pid, scan, sdu, srs)
# find the match from LIDC-IDRI annotation
import math
lunaantdictnodid = {}
maxdist = 0
for srcid, lunaantlidc in lunantdictlidc.items():
	# print('srcid, lunaantlidc',srcid, lunaantlidc)#1.3.6.1.4.1.14519.5.2.1.6279.6001.752756872840730509471096155114 [[[56.39315434, 67.68008675, -64.67445255, 19.65387738], array([108.23021898, 327.93611484,  18.76978102])]]
	lunaantdictnodid[srcid] = []
	pid, stdid = sidmap[srcid]
	# print('pid, stdid',pid, stdid)#LIDC-IDRI-0172 1.3.6.1.4.1.14519.5.2.1.6279.6001.481620140149228611720235499832
	pid = pid[len('LIDC-IDRI-'):]
	# print('pid, stdid',pid, stdid)#0172 1.3.6.1.4.1.14519.5.2.1.6279.6001.481620140149228611720235499832
	for lunantdictlidcsub in lunaantlidc:
		lunaant = lunantdictlidcsub[0]#[56.39315434, 67.68008675, -64.67445255, 19.65387738]
		voxcrd = lunantdictlidcsub[1] # z y x array([108.23021898, 327.93611484,  18.76978102])
		mindist, minidx = 1e8, -1
		if srcid in ['1.3.6.1.4.1.14519.5.2.1.6279.6001.174692377730646477496286081479', '1.3.6.1.4.1.14519.5.2.1.6279.6001.300246184547502297539521283806']:
			continue
		# print('pid+srcid]',pid+'_'+srcid)#0172_1.3.6.1.4.1.14519.5.2.1.6279.6001.752756872840730509471096155114
		for idx, lidcant in enumerate(antdictscan[pid+'_'+srcid]):
			# print('0',pid)#0748
			# print('0',pid+'_'+srcid)
			# print('idx, lidcant',idx, lidcant)#6 [7, 41.53, 4.297, 106, 155, 74, 11, 'Nodule 001', 68972, 69612]
			dist = math.pow(voxcrd[0] - lidcant[5], 2) # z
			dist = math.pow(voxcrd[1] - lidcant[4], 2) # y
			dist += math.pow(voxcrd[2] - lidcant[3], 2) # x
			if dist < mindist:
				mindist = dist
				minidx = idx
		if mindist > 71:#15.1:
			print('71',srcid, pid, voxcrd, antdictscan[pid+'_'+srcid], mindist)
		maxdist = max(maxdist, mindist)
		lunaantdictnodid[srcid].append([lunaant, antdictscan[pid+'_'+srcid][minidx][6:]])
np.save('lunaantdictnodid.npy', lunaantdictnodid)
print('maxdist', maxdist)

# print(stop)
# save it into a csv
import csv
savename = 'annotationnodid.csv'
fid = open(savename, 'w')
writer = csv.writer(fid)
writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'])
for srcid, ant in lunaantdictnodid.items():
	for antsub in ant:
		writer.writerow([srcid] + [antsub[0][0], antsub[0][1], antsub[0][2], antsub[0][3]] + antsub[1])
fid.close()
# find the malignancy, shape information from xml file
import xml.dom.minidom
lunadctclssgmdict = {}
for srsid, extant in lunaantdictnodid.items():
	lunadctclssgmdict[srsid] = []
	pid, stdid = sidmap[srsid]
	for extantvlu in extant:
		mallst, callst, sphlst, marlst, loblst, spilst, texlst = [], [], [], [], [], [], []
		for fname in os.listdir(os.path.join(*['/home/zhaojie/zhaojie/Lung/data/LIDC-IDRI/', pid, stdid, srsid])):
			if fname.endswith('.xml'):
				dom = xml.dom.minidom.parse(os.path.join(*['/home/zhaojie/zhaojie/Lung/data/LIDC-IDRI/', pid, stdid, srsid, fname]))
				root = dom.documentElement
				rsessions = root.getElementsByTagName('readingSession')
				for rsess in rsessions:
					unblinds = rsess.getElementsByTagName('unblindedReadNodule')
					for unb in unblinds:
						nod = unb.getElementsByTagName('noduleID')
						if len(nod) != 1: continue
						if nod[0].firstChild.data in extantvlu[1]:
							cal = unb.getElementsByTagName('calcification')
							# print cal[0].firstChild.data, range(1,7,1), int(cal[0].firstChild.data) in range(1,7,1)
							if len(cal) == 1 and int(cal[0].firstChild.data) in range(1, 7, 1):
								callst.append(float(cal[0].firstChild.data))
							sph = unb.getElementsByTagName('sphericity')
							if len(sph) == 1 and int(sph[0].firstChild.data) in range(1, 6, 1):
								sphlst.append(float(sph[0].firstChild.data))
							mar = unb.getElementsByTagName('margin')
							if len(mar) == 1 and int(mar[0].firstChild.data) in range(1, 6, 1):
								marlst.append(float(mar[0].firstChild.data))
							lob = unb.getElementsByTagName('lobulation')
							if len(lob) == 1 and int(lob[0].firstChild.data) in range(1, 6, 1):
								loblst.append(float(lob[0].firstChild.data))
							spi = unb.getElementsByTagName('spiculation')
							if len(spi) == 1 and int(spi[0].firstChild.data) in range(1, 6, 1):
								spilst.append(float(spi[0].firstChild.data))
							tex = unb.getElementsByTagName('texture')
							if len(tex) == 1 and int(tex[0].firstChild.data) in range(1, 6, 1):
								texlst.append(float(tex[0].firstChild.data))
							mal = unb.getElementsByTagName('malignancy')
							if len(mal) == 1 and int(mal[0].firstChild.data) in range(1, 6, 1):
								mallst.append(float(mal[0].firstChild.data))
		vlulst = [srsid, extantvlu[0][0], extantvlu[0][1], extantvlu[0][2], extantvlu[0][3]]
		if len(mallst) == 0: vlulst.append(0)
		else: vlulst.append(sum(mallst)/float(len(mallst)))
		if len(callst) == 0: vlulst.append(0)
		else: vlulst.append(sum(callst)/float(len(callst)))
		if len(sphlst) == 0: vlulst.append(0)
		else: vlulst.append(sum(sphlst)/float(len(sphlst)))
		if len(marlst) == 0: vlulst.append(0)
		else: vlulst.append(sum(marlst)/float(len(marlst)))
		if len(loblst) == 0: vlulst.append(0)
		else: vlulst.append(sum(loblst)/float(len(loblst)))
		if len(spilst) == 0: vlulst.append(0)
		else: vlulst.append(sum(spilst)/float(len(spilst)))
		if len(texlst) == 0: vlulst.append(0)
		else: vlulst.append(sum(texlst)/float(len(texlst)))
		lunadctclssgmdict[srsid].append(vlulst)
		# print('lunadctclssgmdict.npy', lunadctclssgmdict[srsid])#[['1.3.6.1.4.1.14519.5.2.1.6279.6001.229860476925100292554329427970', -108.0795881, -70.76676061, -138.1757851, 4.341588419, 1.0, 6.0, 5.0, 5.0, 1.0, 1.0, 5.0], ['1.3.6.1.4.1.14519.5.2.1.6279.6001.229860476925100292554329427970', -113.1622011, 35.11715185, -166.5081249, 4.635624615, 2.0, 6.0, 4.0, 5.0, 1.0, 1.0, 5.0]]
		# lunadctclssgmdict[srsid].append([extantvlu[0][0], extantvlu[0][1], extantvlu[0][2], extantvlu[0][3]]+\
		# 	[sum(mallst)/float(len(mallst)), sum(callst)/float(len(callst)), sum(sphlst)/float(len(sphlst)), \
		# 	sum(marlst)/float(len(marlst)), sum(loblst)/float(len(loblst)), sum(spilst)/float(len(spilst)), \
		# 	sum(texlst)/float(len(texlst))])
np.save('lunadctclssgmdict.npy', lunadctclssgmdict)

savename = 'annotationdetclssgm.csv'
fid = open(savename, 'w')
writer = csv.writer(fid)
writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant', 'calcification', \
	'sphericity', 'margin', 'lobulation', 'spiculation', 'texture'])
for srsid, extant in lunadctclssgmdict.items():
	for subextant in extant:
		writer.writerow(subextant)
fid.close()
# discrete the generated csv
import pandas as pd 
import csv
srcname = 'annotationdetclssgm.csv'
dstname = 'annotationdetclssgmfnl.csv'
colname = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant', 'calcification', 'sphericity', \
    'margin', 'lobulation', 'spiculation', 'texture']
srcframe = pd.read_csv(srcname, names=colname)
srslist = srcframe.seriesuid.tolist()[1:]
cdxlist = srcframe.coordX.tolist()[1:]
cdylist = srcframe.coordY.tolist()[1:]
cdzlist = srcframe.coordZ.tolist()[1:]
dimlist = srcframe.diameter_mm.tolist()[1:]
mlglist = srcframe.malignant.tolist()[1:]
callist = srcframe.calcification.tolist()[1:]
sphlist = srcframe.sphericity.tolist()[1:]
mrglist = srcframe.margin.tolist()[1:]
loblist = srcframe.lobulation.tolist()[1:]
spclist = srcframe.spiculation.tolist()[1:]
txtlist = srcframe.texture.tolist()[1:]
fid = open(dstname, 'w')
writer = csv.writer(fid)
writer.writerow(colname)
for idx in range(len(srslist)):
	lst = [srslist[idx], cdxlist[idx], cdylist[idx], cdzlist[idx], dimlist[idx]]
	if abs(float(mlglist[idx]) - 0) < 1e-2: # 0 1 2
		lst.append(0)
	elif abs(float(mlglist[idx]) - 3) < 1e-2:
		lst.append(0)
	elif float(mlglist[idx]) > 3:
		lst.append(2)
	else:
		lst.append(1)
	lst.append(int(round(float(callist[idx])))) # 0 - 6
	if abs(float(sphlist[idx]) - 0) < 1e-2: # 0 1 2 3
		lst.append(0)
	elif float(sphlist[idx]) < 2:
		lst.append(1)
	elif float(sphlist[idx]) < 4:
		lst.append(2) 
	else:
		lst.append(3) 
	if abs(float(mrglist[idx]) - 0) < 1e-2: # 0 1 2
		lst.append(0)
	elif float(mrglist[idx]) < 3:
		lst.append(1)
	else:
		lst.append(2)
	if abs(float(loblist[idx]) - 0) < 1e-2: # 0 1 2
		lst.append(0)
	elif float(loblist[idx]) < 3:
		lst.append(1)
	else:
		lst.append(2)
	if abs(float(spclist[idx]) - 0) < 1e-2: # 0 1 2
		lst.append(0)
	elif float(spclist[idx]) < 3:
		lst.append(1)
	else:
		lst.append(2)
	if abs(float(txtlist[idx]) - 0) < 1e-2: # 0 1 2 3
		lst.append(0)
	elif float(txtlist[idx]) < 2:
		lst.append(1)
	elif float(txtlist[idx]) < 4:
		lst.append(2)
	else:
		lst.append(3)
	writer.writerow(lst)
fid.close()

# fuse annotations for different nodules, generate patient level annotation
import pandas as pd 
import csv
antpdframe = pd.read_csv('annotationdetclssgmfnl.csv', names=['seriesuid', 'coordX', 'coordY', 'coordZ', \
	    'diameter_mm', 'malignant', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture'])
srslst = antpdframe.seriesuid.tolist()[1:]
cdxlst = antpdframe.coordX.tolist()[1:]
cdylst = antpdframe.coordY.tolist()[1:]
cdzlst = antpdframe.coordZ.tolist()[1:]
mlglst = antpdframe.malignant.tolist()[1:]
dimlst = antpdframe.diameter_mm.tolist()[1:]
clclst = antpdframe.calcification.tolist()[1:]
sphlst = antpdframe.sphericity.tolist()[1:]
mrglst = antpdframe.margin.tolist()[1:]
loblst = antpdframe.lobulation.tolist()[1:]
spclst = antpdframe.spiculation.tolist()[1:]
txtlst = antpdframe.texture.tolist()[1:]
dctdat = {}
for idx, srs in enumerate(srslst):
	if mlglst[idx] == '0':
		continue
	vlu = [mlglst[idx], clclst[idx], sphlst[idx], mrglst[idx], loblst[idx], spclst[idx], txtlst[idx]]
	if srs not in dctdat:
		dctdat[srs] = [vlu]
	else:
		dctdat[srs].append(vlu)
fid = open('annotationdetclssgmv2.csv', 'w')
writer = csv.writer(fid)
writer.writerow(['seriesuid', 'malignant'])
for srs, vlulst in dctdat.items():
	mlg = -1
	for vlu in vlulst:
		# print('0',mlg, int(vlu[0]))
		mlg = max(mlg, int(vlu[0]))
	writer.writerow([srs, mlg])
fid.close()
