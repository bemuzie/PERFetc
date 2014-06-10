import nibabel as nib
import sys
import os
import numpy as np
from scipy import stats
class Vividict(dict):

    def __missing__(self, key):    	
        value = self[key] = type(self)()
        return value

class Roi():
	def __init__(self):
		#loading rois in folder
		#forming vol name = roi_name - roi*
		
		self.rois=Vividict()


	def save_txt(self,fname,data_folder=None):
		"""
		Save ROIs data in txt
		
		txt column names are:
			roi: name of ROI
			times : array of times of series included in any roi of Roi.object, separated with ';'
			data x : path to data file of serie with time x, wich was saved with np.savetxt
			series mean_density x :  mean density of ROI in time x
			series median_density x :  median density of ROI in time x
			series sd x :  mean SD of ROI in time x

	    Args:
	      fname (str): filename
	      data_folder (str): folder roi np.arrays be saved. If None they will be saved in the same folder as file.

	    Returns:
	      
    
    
		"""
		out_file = open(fname,'w')

		times = self.get_time_list()
		headers = ['roi', 'times']
		headers += ['data %s'%i for i in times]
		headers += ['series mean_density %s'%i for i in times]
		headers += ['series median_density %s'%i for i in times]
		headers += ['series sd_density %s'%i for i in times]
		#write headers
		out_file.write(','.join(headers))
		out_file.write('\n')
		print 'datafolder',data_folder
		#write data
		for r_name in self.rois:

			line = [r_name]
			line += ';'.join(map(str,times)),
			for t in times:
				print r_name,t,self.rois[r_name]['series'][t]['data']
				try:
					np.savetxt(data_folder+'/%s_%s.txt'%(r_name,t), self.rois[r_name]['series'][t]['data'])
					print os.path.join(data_folder,'%s_%s.txt'%(r_name,t))
					line += os.path.join(data_folder,'%s_%s.txt'%(r_name,t)),
				except IndexError as s:
					if s[0]=="tuple index out of range":
						line += '',
					else:
						raise IndexError,s



			line +=[str(self.rois[r_name]['series'][i]['mean_density']) for i in times]
			line +=[str(self.rois[r_name]['series'][i]['median_density']) for i in times]
			line +=[str(self.rois[r_name]['series'][i]['sd_density']) for i in times]
			
			out_file.write( ','.join(line) )
			out_file.write('\n')

		out_file.close()

			

	def get_time_list(self):
		times={}
		for r_name in self.rois:
			for ts in self.rois[r_name]['series']:
				times[ts]=0
		
		return sorted([i for i in times])

	def update_series(self,r_name=None):
		if not r_name:
			r_name = [i for i in self.rois]
		for i in r_name:
			for t in self.get_time_list():
				d = self.rois[i]['series'][t]['data']
				print i,t
				try:
					self.rois[i]['series'][t]['mean_density'] = np.mean(d)
					self.rois[i]['series'][t]['median_density'] = np.median(d)
					self.rois[i]['series'][t]['sd_density'] = np.std(d)
				except TypeError as s:
					if s[0] == "unsupported operand type(s) for /: 'Vividict' and 'float'":
						pass
					else:
						print s,type(s),s[0]
						raise TypeError, s



	def update_summary(self):
		for i in self.rois:
			d,t = max([[v['mean_d'],t] for t,v in self.rois[i]['series'].items()])
			self.rois[i]['sum']['max_d'] = d
			self.rois[i]['sum']['max_t'] = t
			



	def load(self,path):
		f = open(path)
		headers = f.readline()
		headers = headers.strip('\n').split(',')
		for line in f:
			l_dict = dict([i,ii] for i,ii in zip(headers,line.split(',')))
			r_name = l_dict['roi']

			for n,v in l_dict.items():
				col_name = n.split(' ')
				if col_name[0] == 'series':
					self.rois [r_name] ['series'] [int(col_name[-1])] [col_name[1]] = float(v)
				elif col_name[0] == 'data':
					try:
						self.rois [r_name] ['series'] [int(col_name[-1])] ['data'] = np.loadtxt(os.path.join(os.path.v))
					except:
						continue
					
				else:
					continue
		self.update_series()
		self.update_summary()



	def add_roi_from_file(self,roi_name,roi_file,vol_file,vol_time,cut_and_save=None,update_info=True):
		print vol_file
		vol_np = nib.load(vol_file).get_data()
		roi_np = nib.load(roi_file).get_data()[...,0]
		data = vol_np[roi_np>0]
		self.rois[roi_name]['series'][int(vol_time)]['data'] = data
		if update_info:
			self.update_series()
			self.update_summary()


	def add_rois_from_csv(self,roi_csv):
		f = open(roi_csv)
		for line in f:
			print line
			if line == '\n':
				line = f.next()
				
				roi_name,roi_file = line.strip('\n').split(',')
			else: 
				vol_file,voi_time = line.strip('\n').split(',')
				print roi_name,voi_time
				self.add_roi_from_file(roi_name, roi_file, vol_file, voi_time, update_info=False)
		f.close()
		self.update_series()
		self.update_summary()

		


	def __read_info(self,rois_info):
		f=open(rois_info,'r')
		
		r_folder = f.readline()[:-1]
		self.vol_names = {}
		
		for i in f.readlines():
			ii=i.split(',')
			self.vol_names[os.path.join(r_folder,ii[0])] = int(ii[1])

	def __get_concentrations(self):
		self.concentrations=[]
		self.times=[]
		for vol_path, vol_time in self.vol_names.items():
			vol_f = nib.load(os.path.abspath(vol_path))
			vol = vol_f.get_data()
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

	def get_concentrations(self,r_name):
		
		return [self.rois[r_name]['series'][t]['mean_density'] for t in self.get_time_list()]
	def export_disserdb(self,exam_num,db_path=''):
		sys.path.append(db_path)
		from django.core.management import setup_environ
		from django.core.exceptions import ObjectDoesNotExist

		import BigTable.settings
		setup_environ(BigTable.settings)
		from exams.models import Patient,Examination,Perfusion,Density

		examination = Examination.objects.get(pk=exam_num)
		for t in self.get_time_list():
			ph=examination.phase_set.get(time=t)
			for rname in self.rois:
				try:
					r=ph.density_set.get(roi=rname)
					r.density = self.rois[r_name]['series'][t]['median_density']
					r.save()
				except ObjectDoesNotExist:
					print self.rois[rname]['series'][t]['median_density']
					print type(self.rois[rname]['series'][t]['median_density'])
					ph.density_set.add(Density(roi=rname,density=int(self.rois[rname]['series'][t]['median_density'])))



if __name__ == "__main__":
	temnosagatij = Roi()
	temnosagatij.add_rois_from_csv('/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/aorta_info.csv')
	print temnosagatij.get_concentrations('aorta')
	temnosagatij.save_txt('/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/1/roi_test.csv','/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/1/')
	print temnosagatij.get_concentrations('aorta')
	temnosagatij.load('/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/1/roi_test.csv')
	print temnosagatij.get_concentrations('aorta')
	temnosagatij.save_txt('/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/1/roi_test.csv','/home/denest/PERF_volumes/TEMNOSAGATYI  A.V. 02.04.1973/ROI/1/')
	print temnosagatij.get_concentrations('aorta')