#-*- coding: utf8 -*-
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.integrate import quadrature
import nibabel as nib


def get_tac(roi, vol4d, time):
	tac=[]
	for t in range(len(time)):
		tac +=np.median(vol4d[...,t][roi==1]),
	return np.array(tac)

def tdc_create(input_conc,time):
	windows = np.arange(10)
	volumes = np.arange(0.1,1,0.1)

def run_sum(input_conc,time,window):
	cm=np.concatenate([np.ones(window),np.zeros(window-1)])/float(window)
	print cm
	return np.convolve(input_conc,cm)
def run_trapz(input_conc,time,window):
	o=np.zeros(input_conc.shape)
	scale = np.max(input_conc)
	for i in range(len(input_conc)):
		fr = (i-window>0) and i-window or 0
		o[i] = np.trapz(input_conc[fr:i])/window
	return o
def spline_interpolation(conc,time,new_time,ss=0):
	tck=interpolate.splrep(time,conc,s=ss)
	return interpolate.splev(new_time,tck,der=0),tck

def get_tissue_tac_mrx(aorta_spline,mtt,bv,t):
	t = np.array(t)
	final_arr=[]
	
	for step in mtt:
		from_t = t-step
		from_t[from_t<np.min(t)] = np.min(t)
		final_arr += [np.array([interpolate.splint(ft,tt,aorta_spline)/step for ft,tt in zip(from_t,t)])],
		
	final_arr=np.vstack((final_arr))
	final_arr=final_arr[:,:,None]*bv
	
	return final_arr

def get_tissue_tac(aorta_spline,mtt,bv,t):
	t = np.array(t)
	final_arr=[]
	

	from_t = t-mtt
	from_t[from_t<t[0]] = t[0]
	
	final_arr = np.array([interpolate.splint(ft,tt,aorta_spline)/mtt for ft,tt in zip(from_t,t)])
	final_arr*=bv
	
	return final_arr


def tac_mrx(input_tac,t,mtt_range,bv_range):
	#smothing input curve
	tck = interpolate.splrep(t,input_tac,s=0)


def calculate_mtt_bv(tac,example_mrx,mtt_vector,bv_vector):
	#tac=np.array(tac)
	#print tac.shape,example_mrx.shape
	diff = (example_mrx-tac[None,:,None])
	ssd=np.sum(diff*diff,axis=1)
	mtt_idx,bv_idx=np.where(ssd==np.min(ssd))
	return mtt_vector[mtt_idx],bv_vector[bv_idx]

def make_map(vol4d,input_tac,mtt_range,bv_range,t):
	
	bv_vol = np.zeros(vol4d.shape[:-1])
	mtt_vol = np.zeros(vol4d.shape[:-1])
	mtt=np.arange(mtt_range[0],mtt_range[1],mtt_range[2])
	bv=np.arange(bv_range[0],bv_range[1],bv_range[2])
	
	for m in mtt:
		for b in bv:
			tac=get_tissue_tac(input_tac,m,b,t)
			diff=vol4d-tac

			diff=diff*diff
			ssd_vol2=np.sum(diff,axis=3)

			try:
				mask_array = ssd_vol1>=ssd_vol2

			except NameError:
				mask_array=np.ones(ssd_vol2.shape,dtype=np.bool)
				ssd_vol1=np.copy(ssd_vol2)
			
			bv_vol[mask_array]=b
			mtt_vol[mask_array]=m
			ssd_vol1[mask_array]=ssd_vol2[mask_array]

	bv_vol*=100 # Blood volume should be in ml/100ml not percent
	#mtt_vol/=60. # MTT in min^-1
	bf_vol = bv_vol/(mtt_vol/60.)
	output_vol = np.concatenate((bv_vol[...,None],mtt_vol[...,None], bf_vol[...,None],ssd_vol1[...,None]),axis=3)
	output_vol[-1] = output_vol[1]/output_vol[0]
	
	return output_vol

def make_map2(vol4d,input_tac,mtt_range,bv_range,t):
	
	bv_vol = np.zeros(vol4d.shape[:-1])
	mtt_vol = np.zeros(vol4d.shape[:-1])

	mtt=np.arange(mtt_range[0],mtt_range[1],mtt_range[2])
	bv=np.arange(bv_range[0],bv_range[1],bv_range[2])
	curves = get_tissue_tac_mrx(s,mtt,bv,t)
	# making arrays of curves,bv,mtts
	


	

	for m in mtt:
		for b in bv:
			tac=get_tissue_tac(input_tac,m,b,t)
			diff=vol4d-tac

			diff=diff*diff
			ssd_vol2=np.sum(diff,axis=3)

			try:
				mask_array = ssd_vol1>=ssd_vol2

			except NameError:
				mask_array=np.ones(ssd_vol2.shape,dtype=np.bool)
				ssd_vol1=np.copy(ssd_vol2)
			
			bv_vol[mask_array]=b
			mtt_vol[mask_array]=m
			ssd_vol1[mask_array]=ssd_vol2[mask_array]

	bv_vol*=100 # Blood volume should be in ml/100ml not percent
	#mtt_vol/=60. # MTT in min^-1
	bf_vol = bv_vol/(mtt_vol/60.)
	output_vol = np.concatenate((bv_vol[...,None],mtt_vol[...,None], bf_vol[...,None],ssd_vol1[...,None]),axis=3)
	output_vol[-1] = output_vol[1]/output_vol[0]
	
	return output_vol

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	
	
	a=[44.64877990864637	,
		60.49938896933365	,
		185.7885957985045	,
		424.99678292042654	,
		521.9514543622375	,
		473.2310834657947	,
		325.303563042722	,
		175.04206749112657	,
		108.29381960598403	,
		85.55330475432997	,
		83.41730958452371	,
		125.78528484932369	,
		114.54445596352669	,
		113.53738966541108	,
		116.15463139449018	,
		117.12762712371725	,
		105.83024818019554	,
		101.09165687256322	,
		94.59840815055142	]


	t=[10	,
		13	,
		15	,
		17	,
		20	,
		22	,
		24	,
		26	,
		29	,
		31	,
		33	,
		42	,
		47	,
		52	,
		57	,
		62	,
		91	,
		101	,
		111]

	print len(t)

	#plt.plot(run_sum(a,1,10))
	a=np.array(a)
	a-=a[0]
	t=np.array(t)


	p = interp1d(t, a, kind='cubic')
	#plt.plot(np.arange(np.min(t),np.max(t),.1),p(np.arange(np.min(t),np.max(t),.1)),color='k')
	#print quadrature(p,[1,2,3],[2,3,4])
	#plt.plot(t,run_trapz(a,1,4),color='red')
	spl,s=spline_interpolation(a,t,np.arange(8,90,.1))

	mtts = np.arange(5,40,1)
	bvs = np.arange(0.1,1,0.1)
	
	pancreas=get_tissue_tac_mrx(s,mtts,bvs,t)
	
	"""
	curves_mrx = np.zeros([len(a),len(range(10))])
	print curves_mrx.shape
	for j in range(10):
		curves_mrx[:,j]=run_trapz(a,1,j)
	plt.plot(curves_mrx,color='red')
	plt.plot(curves_mrx*.2,color='blue')
	
	"""

	plt.plot(t,a)

	path_to4d=u'/home/denest/PERF_volumes/MOSKOVTSEV  V.I. 18.01.1938/20140610_632/NII/FILTERED_TOSHIBA_REG/4D00.nii'
	nii4=nib.load(path_to4d)
	niiData=nii4.get_data()
	#bv_vol=np.zeros(niiData.shape[:-1])
	#mtt_vol=np.zeros(niiData.shape[:-1])
	#out_vol = np.zeros(niiData.shape[:-1]+(3,))
	print niiData[...,0][...,None].shape,niiData.shape
	
	#niiData=np.concatenate((niiData[...,0][...,None],niiData),axis=3)

	"""
	for x in range(niiData.shape[0]):
		for y in range(niiData.shape[1]):
			for z in range(niiData.shape[2]):
				#print niiData[x,y,z].shape
				out_vol[x,y,z,:-1] = calculate_mtt_bv(niiData[x,y,z],pancreas,mtts,bvs)
	out_vol[...,-1] = (out_vol[...,1]*100.)/(out_vol[...,0]/60.)
	"""
	niiData=niiData-niiData[...,0][...,None]
	out_vol=make_map(niiData,s,mtt_range=(1,80,2),bv_range=(0.1,1,0.1),t=t)
	
	nii_im = nib.Nifti1Image(out_vol, nii4.get_header().get_sform(), nii4.get_header())

	nib.nifti1.save(nii_im, 
					u'/home/denest/PERF_volumes/MOSKOVTSEV  V.I. 18.01.1938/20140610_632/NII/FILTERED_TOSHIBA_REG/4D00_bf_map.nii' )
	
	plt.show()

