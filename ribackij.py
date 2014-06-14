# -*- coding: utf-8-*-
import os
import re
import subprocess
from image import nii_separator
from image import dcm_parser
import nibabel as nib
import numpy as np
from filters import bilateral


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

    def add_path(self, path_to, l, s=None, a=None, add_to=None):
        if add_to:
            path_to = os.path.join(add_to, path_to)
        if not os.path.isdir(path_to) and not os.path.isfile(path_to):
            os.makedirs(path_to)

        self.p[l][s][a] = path_to

    def get_path(self, l, s=None, a=None):
        print l, s, a
        return self.p[l][s][a]

    def files_in(self, l, s=None, a=None):

        target_folder = self.get_path(l, s)
        return [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]

    def print_all(self):
        for i, v in self.p.items():
            print i, v

    def gp(self, l, s=None, a=None):
        self.get_path(l, s, a)


CROP_VOLUME = 'crop_volume.nii.gz'


# The workflow for recieved DICOM in some TEMP folder


# Parse DICOM and move them to DATA_STORAGE folder with generated subfolder structure /Patient_name/DCM/Examination_date/Series_Kernel_Filter
def create_folders():
    global pat
    pat = DirManager()
    pat.add_path(ROOT_FOLDER, 'root')
    pat.add_path('DCM', 'dcm', add_to=pat.get_path('root'))
    pat.add_path('NII', 'nii', add_to=pat.get_path('root'))
    pat.add_path('ROI', 'roi', add_to=pat.get_path('root'))
    pat.add_path('NII/RAW/', 'nii_raw', add_to=pat.get_path('root'))
    pat.add_path(SEPARATED, 'separated', add_to=pat.get_path('nii'))

    pat.add_path('crop_volume.nii.gz', 'crop', add_to=pat.get_path('roi'))
    pat.add_path(FILTERED, 'filtered', add_to=pat.get_path('nii'))
    TARGET_PHASE = 10
    pat.add_path('registration_roi.nii.gz', 'registration_roi', add_to=pat.get_path('roi'))
    pat.add_path('aorta_roi.nii.gz', 'aorta', add_to=pat.get_path('roi'))
    pat.add_path('pancreas_roi.nii.gz', 'pancreas', add_to=pat.get_path('roi'))
    pat.add_path('tumor_roi.nii.gz', 'tumor', add_to=pat.get_path('roi'))

    pat.add_path(REGISTERED, 'reg', add_to=pat.get_path('nii'))

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


def convert_dcm_to_nii():
    DCM2NII_PATH = '/home/denest/mricron/dcm2nii'
    ini_path = 'dcm2nii.ini'
    print pat.get_path('nii_raw')
    dcm2nii_pars = ['-4 y',  # Create 4D volumes, else DTI/fMRI saved as many 3D volumes: Y,N = Y
                    '-a n',  # Anonymize [remove identifying information]: Y,N = Y
                    '-b dcm2nii.ini',  # load settings from specified inifile, e.g. '-b C:\set\t1.ini'
                    #'-c y',#Collapse input folders: Y,N = Y
                    '-d n',  #Date in filename [filename.dcm -> 20061230122032.nii]: Y,N = Y
                    '-e y',  # events (series/acq) in filename [filename.dcm -> s002a003.nii]: Y,N = Y
                    '-f n',  # Source filename [e.g. filename.par -> filename.nii]: Y,N = N
                    '-g y',  #gzip output, filename.nii.gz [ignored if '-n n']: Y,N = Y
                    #'-i y',#ID  in filename [filename.dcm -> johndoe.nii]: Y,N = N
                    #'-m n',#manually prompt user to specify output format [NIfTI input only]: Y,N = Y
                    '-n y',  #output .nii file [if no, create .hdr/.img pair]: Y,N = Y
                    "-o '%s'" % pat.get_path('nii_raw'),
                    #Output Directory, e.g. 'C:\TEMP' (if unspecified, source directory is used)
                    '-p n',  #Protocol in filename [filename.dcm -> TFE_T1.nii]: Y,N = Y
                    '-r n',  # Reorient image to nearest orthogonal: Y,N
                    '-s n',  #SPM2/Analyze not SPM5/NIfTI [ignored if '-n y']: Y,N = N
                    '-v y',  #Convert every image in the directory: Y,N = Y
                    '-x n',  #Reorient and crop 3D NIfTI images: Y,N = N
                    "'%s'" % pat.get_path('dcm')  #INPUT FOLDER
    ]
    subprocess.check_call(' '.join(['gzip', '-rd', "'%s'" % pat.get_path('dcm')]),
                          shell=True)
    subprocess.check_call(' '.join([DCM2NII_PATH, ' '.join(dcm2nii_pars)]),
                          shell=True)
    subprocess.check_call(' '.join(['gzip', '-r -1', "'%s'" % pat.get_path('dcm')]),
                          shell=True)


# convert_dcm_to_nii()
# compress DICOMs

#Separate 4d NIIs to 3d NIIs and move them to /Patient_name/NII/(Examination_date)_(Series)/(Examination_date)_(Series)_time.nii.gz



def separate_nii(path_to_nii, output_path):
    for p, d, f in os.walk(path_to_nii):
        for fname in f:
            try:
                nii_separator.separate_nii(os.path.join(p, fname), output_path)
            except ValueError, s:
                if s == 'Expecting four dimensions':
                    continue


#Select crop volume for NIFTies









#Filter 3dNIIs with 3d bilateral filter with
#move them to /Patient_name/NII/(Examination_date)_(Series)_filter_I(IntensitySigma)_G(GaussianSigma)/(Examination_date)_(Series)_time_filter_I(IntensitySigma)_G(GaussianSigma).nii.gz





def filter_vols():
    cr_vol = nib.load(pat.get_path('crop')).get_data()
    if len(cr_vol.shape) == 4:
        cr_vol = cr_vol[..., 0]

    borders_vol = np.where(cr_vol == 1)
    x_fr, y_fr, z_fr = map(np.min, borders_vol)
    x_to, y_to, z_to = map(np.max, borders_vol)
    print x_fr, y_fr, z_fr, x_to, y_to, z_to

    for p, d, f in os.walk(pat.get_path('separated')):
        for fname in f:
            INTENSITY_SIGMA = 40
            GAUSSIAN_SIGMA = 1.5
            nii_separator.set_header_from(os.path.join(p, fname),
                                          '/home/denest/PERF_volumes/ZAKHAROVA  O.A. 13.11.1981/20140610_635/NII/SEPARATED/s004a001_3.nii.gz')
            if not os.path.isfile(os.path.join(pat.get_path('filtered'), '%s_I%s_G%s.nii' % (
                    fname.rstrip('.nii.gz'), INTENSITY_SIGMA, GAUSSIAN_SIGMA))):
                bilateral.bilateral(os.path.join(p, fname), output_folder=pat.get_path('filtered'),
                                    sig_i=INTENSITY_SIGMA, sig_g=GAUSSIAN_SIGMA, x_range=[x_fr, x_to],
                                    y_range=[y_fr, y_to], z_range=[z_fr, z_to])
            else:
                print 'exists', os.path.join(pat.get_path('filtered'), '%s_I%s_G%s.nii' % (
                    fname.rstrip('.nii.gz'), INTENSITY_SIGMA, GAUSSIAN_SIGMA))

    for fn in pat.files_in('filtered'):
        splited_fname = re.split(r"s|a|_", fn)
        print splited_fname
        s, a = [int(i) for i in (splited_fname[1], splited_fname[3])]

        pat.add_path(fn, 'filtered', s, a, add_to=(pat.get_path('filtered')))


#make /

#Manual manipulations
#Create ROIs for aorta,IVC
#Choose target phase and make registration

#Create ROIs for pancreas,tumor,tumor1
#Create registration mask



def registration_start():
    ANTs_PATH = '/home/denest/ANTs-1.9.x-Linux/bin/'
    #registration
    WORKING_FOLDER = pat.get_path('root')
    #images_folder = pat.get_path('filtered,4')
    fixed_im = pat.get_path('filtered', TARGET_PHASE)
    #moved_im='20140508_100402GeneralBodyPerfusionCopiedTEMNOSAGATYIAV02041973s004a001_20_I40_G1.5-subvolume-scale_1.nii.gz'
    mask = pat.get_path('registration_roi')
    print 'mask', mask
    output_folder = pat.get_path('reg')
    print 'out:', output_folder

    def registration(moved_image, fixed_image, mask, output_folder):

        prefix = '%s_to_%s' % (os.path.basename(moved_image).split('.')[0], os.path.basename(fixed_image).split('.')[0])
        print prefix
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
        #print ' '.join(registration_parametrs)
        subprocess.check_call('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=3', shell=True)
        subprocess.check_call(' '.join([ANTs_PATH + 'antsRegistration', ' '.join(registration_parametrs)]), shell=True,
                              cwd=output_folder)
        subprocess.check_call(' '.join([ANTs_PATH + 'antsApplyTransforms',
                                        '-d 3',
                                        '-r', fixed_image,
                                        '-i', moved_image,
                                        '-t [%s0GenericAffine.mat,0]' % prefix,
                                        '-t [%s1Warp.nii.gz,0]' % prefix,
                                        '-o %s_registered.nii.gz' % prefix])
                              , shell=True, cwd=output_folder)

    for snum in pat.p['filtered']:
        for anum in pat.p['filtered'][snum]:
            if not all([type(i) is int for i in (anum, snum)]):
                continue

            mi = pat.get_path('filtered', snum, anum)
            print 'moved image', mi
            fi = fixed_im
            print 'fixed_image', fi
            prefix = '%s_to_%s' % (os.path.basename(mi).split('.')[0], os.path.basename(fi).split('.')[0])
            if os.path.isfile(os.path.join(pat.get_path('reg'), '%s_registered.nii.gz' % prefix)):
                #print os.path.join(pat.get_path('reg',4),'%s_registered.nii.gz'%prefix)
                pat.add_path('%s_registered.nii.gz' % prefix, 'reg', snum, anum, add_to=pat.get_path('reg', 4))

            else:
                registration(mi, fi, mask, output_folder)
                pat.add_path('%s_registered.nii.gz' % prefix, 'reg', snum, anum, add_to=pat.get_path('reg', 4))
    """
    for aq_n,file_name in pat.gp('filtered',4):
        if not aq_n:
            continue
        fname = os.path.join(images_folder,file_name)
        if not fname==os.path.join(images_folder ,fixed_im):
            registration(os.path.relpath(fname,output_folder),os.path.relpath(fixed_im,output_folder),os.path.relpath(mask,output_folder),output_folder)
    """


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
    contrast apearance time - Время серии, когда в ЗИ-аорта различия с нативом станут статистически значимы (Тест Стьюдента, p<0,05)
    maximum intensity HU - Максимальная денситометрическая плотность в ЗИ-аорта
    maximum intensity time - Время серии, когда в ЗИ-аорта денситометрическая плотность станет максимальной
    equilibrium concentration HU - Денситометрическая плотность, при которой в ЗИ-аорта различия с ЗИ-нпв(ниже уровня почечных артерий) станут статистически не значимы (Тест Стьюдента, p<0,05)
    equilibrium concentration time - Время серии, в которой различия между ЗИ-аорта и ЗИ-нпв(ниже уровня почечных артерий) станут статистически не значимы (Тест Стьюдента, p<0,05)
pancreas
    volume - объём ЗИ(поджелудочная железа)
    HU - Денситометрическая плотность ЗИ(поджелудочная железа)
    contrast apearance time - Время серии, когда в ЗИ(поджелудочная железа) различия с нативом станут статистически значимы (Тест Стьюдента, p<0,05)
    maximum intensity HU - Максимальная денситометрическая плотность в ЗИ(поджелудочная железа)
    maximum intensity time - Время серии, когда в ЗИ(поджелудочная железа) денситометрическая плотность станет максимальной
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
    ROOT_FOLDER = '/home/denest/PERF_volumes/MOSKOVTSEV  V.I. 18.01.1938/20140610_632'
    SEPARATED = 'RAW3'
    FILTERED = 'FILTERED3'
    REGISTERED = 'REGISTERED3'

    create_folders()
    subprocess.check_call(' '.join(['gzip', '-rd ', "'%s'" % pat.get_path('dcm')]),
                          shell=True)
    dcm_parser.get_times(pat.get_path('dcm'), pat.get_path('roi'))
    subprocess.check_call(' '.join(['gzip', '-r -1', "'%s'" % pat.get_path('dcm')]),
                          shell=True)
    #convert_dcm_to_nii()
    #separate_nii(pat.get_path('nii_raw'),pat.get_path('separated'))


    #filter_vols()
    #registration_start()
    pass