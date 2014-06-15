# -*- coding: utf-8-*-
import os
import re
import subprocess
from image import nii_separator
from image import dcm_parser
import nibabel as nib
import numpy as np
from filters import bilateral


class Workflow():
    def __init__(self, root_folder):
        self.folders = {'root_folder': root_folder}
        self.dir_manager = DirManager()
        self.setup_env()
        self.__create_folders()
        self.gzip = 'gzip'
        self.mricron = 'dcm2nii'
        self.ants = 'ants'


    def setup_env(self):
        self.gzip = os.path.abspath('E:\gzip\gzip.exe')
        self.mricron = os.path.abspath('E:\mricron\dcm2nii.exe')
        self.ants = os.path.abspath('C:\Program Files (x86)\ANTS\\bin')

    def __create_folders(self):
        self.dir_manager.add_path(self.folders['root_folder'], 'root')
        self.dir_manager.add_path('DCM', 'dcm', add_to='root')
        self.dir_manager.add_path('NII', 'nii', add_to='root')
        self.dir_manager.add_path('ROI', 'roi', add_to='root')
        self.dir_manager.add_path('NII/RAW/', 'nii_raw', add_to='root')
        self.dir_manager.add_path('SEPARATED', 'separated', add_to='nii')

        self.dir_manager.add_path('crop_volume.nii.gz', 'crop', add_to='roi', create=False)
        self.dir_manager.add_path('FILTERED', 'filtered', add_to='nii')

        self.dir_manager.add_path('registration_roi.nii.gz', 'registration_roi', add_to='roi', create=False)
        self.dir_manager.add_path('aorta_roi.nii.gz', 'aorta', add_to='roi', create=False)
        self.dir_manager.add_path('pancreas_roi.nii.gz', 'pancreas', add_to='roi', create=False)
        self.dir_manager.add_path('tumor_roi.nii.gz', 'tumor', add_to='roi', create=False)

        self.dir_manager.add_path('REGISTERED', 'reg', add_to='nii')

    def convert_dcm_to_nii(self):
        # ini_path = 'dcm2nii.ini'

        dcm_parser.get_times(self.dir_manager.get_path('dcm'), self.dir_manager.get_path('roi'))

        dcm2nii_pars = ['-4 y',  # Create 4D volumes, else DTI/fMRI saved as many 3D volumes: Y,N = Y
                        '-a n',  # Anonymize [remove identifying information]: Y,N = Y
                        '-b dcm2nii.ini',  # load settings from specified inifile, e.g. '-b C:\set\t1.ini'
                        # '-c y',#Collapse input folders: Y,N = Y
                        '-d n',  # Date in filename [filename.dcm -> 20061230122032.nii]: Y,N = Y
                        '-e y',  # events (series/acq) in filename [filename.dcm -> s002a003.nii]: Y,N = Y
                        '-f n',  # Source filename [e.g. filename.par -> filename.nii]: Y,N = N
                        '-g y',  # gzip output, filename.nii.gz [ignored if '-n n']: Y,N = Y
                        # '-i y',#ID  in filename [filename.dcm -> johndoe.nii]: Y,N = N
                        # '-m n',#manually prompt user to specify output format [NIfTI input only]: Y,N = Y
                        '-n y',  # output .nii file [if no, create .hdr/.img pair]: Y,N = Y
                        """-o "%s" """ % self.dir_manager.get_path('nii_raw'),
                        # Output Directory, e.g. 'C:\TEMP' (if unspecified, source directory is used)
                        '-p n',  # Protocol in filename [filename.dcm -> TFE_T1.nii]: Y,N = Y
                        '-r n',  # Reorient image to nearest orthogonal: Y,N
                        '-s n',  # SPM2/Analyze not SPM5/NIfTI [ignored if '-n y']: Y,N = N
                        '-v y',  #Convert every image in the directory: Y,N = Y
                        '-x n',  #Reorient and crop 3D NIfTI images: Y,N = N
                        """ "%s" """ % self.dir_manager.get_path('dcm')  #INPUT FOLDER
                        ]

        subprocess.check_call(' '.join([self.gzip, '-rd', """ "%s" """ % self.dir_manager.get_path('dcm')]),
                              shell=True)
        subprocess.check_call(' '.join([self.mricron, ' '.join(dcm2nii_pars)]),
                              shell=True)
        subprocess.check_call(' '.join([self.gzip, '-r -1', """ "%s" """ % self.dir_manager.get_path('dcm')]),
                              shell=True)

    def separate_nii(self, path_to_nii=None, output_path=None):
        if not path_to_nii:
            path_to_nii = self.dir_manager.get_path('nii_raw')
        if not output_path:
            output_path = self.dir_manager.get_path('separated')

        for p, d, f in os.walk(path_to_nii):
            for fname in f:
                try:
                    print os.path.join(p, fname)
                    nii_separator.separate_nii(os.path.join(p, fname), output_path)
                except ValueError, s:
                    if s == 'Expecting four dimensions':
                        continue

    def filter_vols(self, intensity_sigma=40, gaussian_sigma=1.5):
        cr_vol = nib.load(self.dir_manager.get_path('crop')).get_data()
        if len(cr_vol.shape) == 4:
            cr_vol = cr_vol[..., 0]

        borders_vol = np.where(cr_vol == 1)
        x_fr, y_fr, z_fr = map(np.min, borders_vol)
        x_to, y_to, z_to = map(np.max, borders_vol)

        for p, d, f in os.walk(self.dir_manager.get_path('separated')):
            for fname in f:


                if not os.path.isfile(os.path.join(self.dir_manager.get_path('filtered'), '%s_I%s_G%s.nii' % (
                        fname.rstrip('.nii.gz'), intensity_sigma, gaussian_sigma))):
                    nii_separator.set_header_from(os.path.join(p, fname),
                                                  '/home/denest/PERF_volumes/ZAKHAROVA  O.A. 13.11.1981/20140610_635/NII/SEPARATED/s004a001_3.nii.gz')
                    bilateral.bilateral(os.path.join(p, fname), output_folder=self.dir_manager.get_path('filtered'),
                                        sig_i=intensity_sigma, sig_g=gaussian_sigma, x_range=[x_fr, x_to],
                                        y_range=[y_fr, y_to], z_range=[z_fr, z_to])
                else:
                    print 'exists', os.path.join(self.dir_manager.get_path('filtered'), '%s_I%s_G%s.nii' % (
                        fname.rstrip('.nii.gz'), intensity_sigma, gaussian_sigma))

        for fn in self.dir_manager.files_in('filtered'):
            splited_fname = re.split(r"s|a|_", fn)
            print splited_fname
            s, a = [int(i) for i in (splited_fname[1], splited_fname[3])]

            self.dir_manager.add_path(fn, 'filtered', s, a, add_to=(self.dir_manager.get_path('filtered')))

    def registration_start(self):
        ants = self.ants
        # registration
        def registration(moved_image, fixed_image, mask, output_folder):
            prefix = '%s_to_%s' % (
                os.path.basename(moved_image).split('.')[0], os.path.basename(fixed_image).split('.')[0])

            fixed_image = os.path.relpath(fixed_image, output_folder)
            moved_image = os.path.relpath(moved_image, output_folder)
            registration_parametrs = ['-d', '3',
                                      '--transform', 'Affine[0.75]',
                                      '--metric', 'MI[%s,%s,0.5,32]' % (fixed_image, moved_image),
                                      '--convergence', '[100x100x0,1e-6,5]',
                                      '--shrink-factors', '8x2x1',
                                      '--smoothing-sigmas', '6x2x2vox',
                                      '--use-estimate-learning-rate-once',
                                      '-x %s' % mask,

                                      '--transform', 'SyN[0.75]',
                                      '--metric', 'MI[%s,%s,0.5,32]' % (fixed_image, moved_image),
                                      '--convergence', '[100x0x0,1e-6,5]',
                                      '--shrink-factors', '6x2x1',
                                      '--smoothing-sigmas', '8x2x1vox',
                                      '--use-estimate-learning-rate-once'
                                      '-x %s' % mask,
                                      '-o', prefix
            ]

            subprocess.check_call('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=3', shell=True)
            subprocess.check_call(' '.join([ants + 'antsRegistration', ' '.join(registration_parametrs)]),
                                  shell=True,
                                  cwd=output_folder)
            subprocess.check_call(' '.join([ants + 'antsApplyTransforms',
                                            '-d 3',
                                            '-r', fixed_image,
                                            '-i', moved_image,
                                            '-t [%s0GenericAffine.mat,0]' % prefix,
                                            '-t [%s1Warp.nii.gz,0]' % prefix,
                                            '-o %s_registered.nii.gz' % prefix])
                                  , shell=True, cwd=output_folder)

        for snum in self.dir_manager.p['filtered']:
            for anum in self.dir_manager.p['filtered'][snum]:
                if not all([type(i) is int for i in (anum, snum)]):
                    continue

                mi = self.dir_manager.get_path('filtered', snum, anum)
                fi = self.dir_manager.get_path('filtered', TARGET_PHASE)

                prefix = '%s_to_%s' % (os.path.basename(mi).split('.')[0], os.path.basename(fi).split('.')[0])

                if os.path.isfile(os.path.join(self.dir_manager.get_path('reg'), '%s_registered.nii.gz' % prefix)):
                    # print os.path.join(pat.get_path('reg',4),'%s_registered.nii.gz'%prefix)
                    self.dir_manager.add_path('%s_registered.nii.gz' % prefix, 'reg', snum, anum,
                                              add_to=self.dir_manager.get_path('reg', 4))

                else:
                    registration(mi, fi, self.dir_manager.get_path('registration_roi'), self.dir_manager.get_path('reg'))
                    self.dir_manager.add_path('%s_registered.nii.gz' % prefix, 'reg', snum, anum,
                                              add_to=self.dir_manager.get_path('reg', 4))


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


class DirManager():
    def __init__(self):
        # self.abs_path = root_folder
        # Creatioon of default folder structure
        # self.dcm,self.nii,self.roi = [ os.pathjoin(self.abs_path,sf) for sf in ('DCM','ROI','NII')]
        self.p = AutoVivification()

    def add_path(self, path_to, l, s=None, a=None, add_to=None, create=True):
        #path_to = os.path.abspath(path_to)
        if add_to:

            path_to = os.path.join(self.get_path(add_to), path_to)
        if not os.path.isdir(path_to) and not os.path.isfile(path_to) and create == True:
            os.makedirs(path_to)

        self.p[l][s][a] = path_to

    def get_path(self, l, s=None, a=None):
        #print l, s, a
        return os.path.abspath(self.p[l][s][a])

    def files_in(self, l, s=None, a=None):

        target_folder = self.get_path(l, s)
        return [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]

    def print_all(self):
        for i, v in self.p.items():
            print i, v

    def gp(self, l, s=None, a=None):
        self.get_path(l, s, a)


# The workflow for recieved DICOM in some TEMP folder


# Parse DICOM and move them to DATA_STORAGE folder with generated subfolder structure /Patient_name/DCM/Examination_date/Series_Kernel_Filter


# make information file /Patient_name/DCM/Examination_date/info.txt



# Convert DICOMs to NIFTI, write NIFTI to /Patient_name/NII/(Examination_date)_(Series).nii.gz
"""
dcm2nii HELP
-4 Create 4D volumes, else DTI/fMRI saved as many 3D volumes: Y,N = Y
-a Anonymize [remove identifying information]: Y,N = Y
-b load settings from specified inifile, e.g. '-b C:\sett1.ini'  
-c Collapse input folders: Y,N = Y
-d Date in filename [filename.dcm -> 20061230122032.nii]: Y,N = Y
-e events (series/acq) in filename [filename.dcm -> s002a003.nii]: Y,N = Y
-f Source filename [e.g. filename.par -> filename.nii]: Y,N = N
-g gzip output, filename.nii.gz [ignored if '-n n']: Y,N = Y
-i ID  in filename [filename.dcm -> johndoe.nii]: Y,N = N
-m manually prompt user to specify output format [NIfTI input only]: Y,N = Y
-n output .nii file [if no, create .hdr/.img pair]: Y,N = Y
-o Output Directory, e.g. 'C:\TEMP' (if unspecified, source directory is used)
-p Protocol in filename [filename.dcm -> TFE_T1.nii]: Y,N = Y
-r Reorient image to nearest orthogonal: Y,N 
-s SPM2/Analyze not SPM5/NIfTI [ignored if '-n y']: Y,N = N
-v Convert every image in the directory: Y,N = Y
-x Reorient and crop 3D NIfTI images: Y,N = N
"""


# convert_dcm_to_nii()
# compress DICOMs

# Separate 4d NIIs to 3d NIIs and move them to /Patient_name/NII/(Examination_date)_(Series)/(Examination_date)_(Series)_time.nii.gz






# Select crop volume for NIFTies









# Filter 3dNIIs with 3d bilateral filter with
# move them to /Patient_name/NII/(Examination_date)_(Series)_filter_I(IntensitySigma)_G(GaussianSigma)/(Examination_date)_(Series)_time_filter_I(IntensitySigma)_G(GaussianSigma).nii.gz






# make /

# Manual manipulations
# Create ROIs for aorta,IVC
# Choose target phase and make registration

# Create ROIs for pancreas,tumor,tumor1
# Create registration mask






"""

#calculte rois parametrs

"""


def roi_calculation():
    roi.add('aorta', pat.get_path('aorta'))
    roi.add('tumor', pat.gp('tumor'))
    roi.add('pancreas', pat.gp('pancreas'))
    for i, t in zip(range(21), time_list):
        for roi_name in ("aorta", "pancreas", "tumor"):
            roi.add_roi_from_file(roi_name, pat.gp(roi_name), pat.gp('reg', 4, i), t)


"""
Subfolder structure
Patient_name/
    |
    |---- DCM/
    |		|----Examination_date/
    |		|		|----Series_Kernel_Filter/
    |		|
    |		|----info.txt
    |
    |---- NII/
    |		|----ExaminationDate_Series_3d/
    |		|----ExaminationDate_Series_3d_filtered_I(IntensitySigma)_G(GaussianSigma)/
    |		|----PatientName_ExaminationDate_Series.nii.gz
    |
    |---- ROI/
    |		|----ExaminationDate_Series_3d/
    |		|----ExaminationDate_Series_3d_filtered_I(IntensitySigma)_G(GaussianSigma)/
    |		|----PatientName_ExaminationDate_Series.nii.gz
    |		|----roi.info
    |
    |----docs/
    |
    |----pics/
    |
    |----reports/
            |----Examination_date_id_report.txt
"""

#Structure
"""
Examination_date_id_report.txt 

ROI
aorta
    contrast apearance time - Время серии, когда в ЗИ-аорта
        различия с нативом станут статистически значимы (Тест Стьюдента, p<0,05)
    maximum intensity HU - Максимальная денситометрическая плотность в ЗИ-аорта
    maximum intensity time - Время серии, когда в ЗИ-аорта денситометрическая плотность станет максимальной
    equilibrium concentration HU - Денситометрическая плотность, при которой в ЗИ-аорта различия с
        ЗИ-нпв(ниже уровня почечных артерий) станут статистически не значимы (Тест Стьюдента, p<0,05)
    equilibrium concentration time - Время серии, в которой различия между ЗИ-аорта и
        ЗИ-нпв(ниже уровня почечных артерий) станут статистически не значимы (Тест Стьюдента, p<0,05)
pancreas
    volume - объём ЗИ(поджелудочная железа)
    HU - Денситометрическая плотность ЗИ(поджелудочная железа)
    contrast appearance time - Время серии, когда в ЗИ(поджелудочная железа)
        различия с нативом станут статистически значимы (Тест Стьюдента, p<0,05)
    maximum intensity HU - Максимальная денситометрическая плотность в ЗИ(поджелудочная железа)
    maximum intensity time - Время серии, когда в ЗИ(поджелудочная железа)
        денситометрическая плотность станет максимальной
    equilibrium concentration HU - Денситометрическая плотность ЗИ(поджелудочная железа) в equilibrium concentration time
tumor
    volume - объём ЗИ(опухоль)
    max size - максимальный размер зоны интереса опухоль
    HU - Денситометрическая плотность ЗИ(опухоль)

    contrast apearance time - Время серии, когда в ЗИ(опухоль) различия с нативом станут статистически значимы (Тест Стьюдента, p<0,05)
    maximum intensity HU - Максимальная денситометрическая плотность в ЗИ(опухоль)
    maximum intensity time - Время серии, когда в ЗИ(опухоль) денситометрическая плотность станет максимальной
    maximum intensity diff HU - Максимальная разница между денситометрической плотностью в ЗИ(опухоль) и ЗИ(поджелудочная железа)
    maximum intensity diff time - Время серии максимальной разницы между денситометрической плотностью в ЗИ(опухоль) и ЗИ(поджелудочная железа)
    equilibrium concentration HU - Денситометрическая плотность ЗИ(опухоль) в equilibrium concentration time



"""

#roi.info
"""
roi_name;vol_path,vol_time....
"""

if __name__ == "__main__":
    ROOT_FOLDER = 'E:/_PerfDB/ROGACHEVSKIJ/ROGACHEVSKIJ V.F. 10.03.1945/20111129_1396'

    wf = Workflow(ROOT_FOLDER)
    #wf.convert_dcm_to_nii()
    wf.separate_nii()

    #convert_dcm_to_nii()
    #separate_nii(pat.get_path('nii_raw'),pat.get_path('separated'))


    #filter_vols()
    #registration_start()
    pass