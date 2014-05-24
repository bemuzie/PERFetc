import nibabel as nib
import os
import numpy as np
from scipy import stats

def create_rois_from_csv(roi_csv):
	f = open(roi_csv)
	rois = {}
	for line in f.readlines():
		if line == '\n':
			new_roi_start = 1
			next(f.readlines())
			roi_name,roi_folder = line[:-2].split(',')
			rois[roi_name] = {'volumes':{}}
		else: 
			rois[roi_name]['volumes'] = 


	pass



class Roi():
	def __init__(self,roi_info_file,rois_info=None):
		#loading rois in folder
		#forming vol name = roi_name - roi*
		f = nib.load(os.path.abspath(roi_file))
		self.roi_vol = f.get_data()[...,0]
		if not rois_info:
			root = os.path.abspath( os.path.join(rois_folder,'..','ROI') )
			rois_info = os.path.join(root, os.path.basename(rois_folder), '_info.txt')
		self.__read_info(rois_info)
		self.__get_concentrations()

		self.mean_concentrations = [np.average(i) for i in self.concentrations]
		self.sd_concentrations = [np.std(i) for i in self.concentrations]
		self.max_hu = np.max(self.mean_concentrations)
		self.max_t = self.times[self.mean_concentrations.index(self.max_hu)]
		self.rois = {'aorta':{'series':{},'sum':{}}}


	def save(self,path):
		headers = ['roi name','data_path','max_hu','max_t']
		times = self.get_time_list()
		#write headers summary data
		out_file.write('roi name','data concentration','times')
		#write headers series data
		out_file.write( ','.join( ['mean %s'%i for i in times] ))
		out_file.write('\n')

		for r_name in self.rois:
			if r_name == concentration:
				np.savetxt()
			out_file.write( [str(self.rois[r_name]['series'][i]['mean_concentration']) for i in times] )

			out_file.write('\n')


			

		np.savetxt('.csv',)
	def get_time_list(self):
		times={}
		for r_name in self.rois:
			for ts in self.rois[r_name]['series']:
				times[ts]=0
		return sorted([i for i in times])
			


	def load(self,path):
		l = dict([[n,i]for n,i in zip(headers,line.split(','))])
		self.rois[l['roi name']] = {'series':dict([[int(i),None] for i in l['times'].split(';')] ),'sum':{} }
		for n,v in l.items():
			col_name = n.split(' ')
			if col_name[0] = 'series':
				self.rois [l['roi name']] ['series'] [col_name[-1]] [col_name[1]] = v
			elif col_name[0] = 'sum':
				self.rois [l['roi name']] ['sum'] [col_name[1]] = v
			elif:
				col_name[0] = 'data':
				self.rois [l['roi name']] ['series'] [col_name[-1]] [col_name[1]] = np.loadtxt(os.path.abspath(v))
			else:
				continue



		
		roi_name = 


	def add_roi_from_file(self,roi_name,roi_file,vol_file,vol_time,cut_and_save=None):

		if cut_and_save:
			vol_np[roi_np>0]=-2047
			nib.save()

		vol_np = nib.load(vol_file).get_data()
		roi_np = nib.load(roi_file).get_data()
		self.rois[roi_name][vol][vol_time] = vol_np[roi_np>0]

		


	def add_rois_from_csv(self,roi_csv):
		f = open(roi_csv)
		for line in f.readlines():
		if line == '\n':
			new_roi_start = 1
			next(f.readlines())
			roi_name,roi_folder = line[:-2].split(',')
			roi_file = os.path.join( os.path.abspath(roi_csv), roi_name, '.nii.gz' )
		else: 
			vol_file,voi_time = line[:-2].split(',')
			self.add_roi(roi_name, roi_file, os.path.join(roi_folder,vol_file), voi_time)

		


	def __read_info(self,rois_info):
		f=open(rois_info,'r')
		
		r_folder = f.readline()[:-1]
		self.vol_names = {}
		
		for i in f.readlines():
			ii=i.split(',')
			print 1
			print int(ii[1])
			self.vol_names[os.path.join(r_folder,ii[0])] = int(ii[1])

	def __get_concentrations(self):
		self.concentrations=[]
		self.times=[]
		for vol_path, vol_time in self.vol_names.items():
			vol_f = nib.load(os.path.abspath(vol_path))
			vol = vol_f.get_data()
			print vol.shape
			print self.roi_vol.shape
			self.concentrations += vol[self.roi_vol==1],
			self.times += vol_time,
		self.concentrations = [c for (t,c) in sorted(zip(self.times, self.concentrations))]
		self.times = sorted(self.times)

	def equilibrium_time(self,aorta_volumes,ivc_volumes):
		aorta_vs_ivc = []
		for a,v in zip(aorta_volumes,ivc_volumes):
			aorta_vs_ivc+=stats.ttest_ind(a,v)[0],
		return aorta_vs_ivc

	def output(self):
		print self.mean_concentrations
		print self.sd_concentrations
		print self.max_hu
		print self.max_t


