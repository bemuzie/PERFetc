#!/usr/bin/env python 

import sys
sys.path.insert(0, '/home/denest/PERFetc2/')

import workflow
from perfusion.image import roi_map
import nibabel as nib

if __name__ == '__main__':
    ROOT_FOLDER_LIN = './'
    wf = workflow.Workflow(ROOT_FOLDER_LIN)
    wf.dir_manager.add_path('FILTERED', 'filtered', add_to='nii')
    wf.setup_env(mricron='/home/denest/mricron/dcm2nii')
    wf.dir_manager.add_path('tumor2.nii.gz', 'tumor', add_to='roi', create=False)

    tum_dist_map =
    tum_ct =

    tum_dist_map[tum_dist_map>0]
    tum_ct[tum_dist_map>0]


    #wf.filter_vols(intensity_sigma=40, gaussian_sigma=1.5)
    #wf.update_label()
    #wf.registration_start(11)
    #wf.make_4dvol()
    #wf.add_roi('aorta')
    #wf.add_roi('porta')
    #wf.add_roi('4d_mask')
    """
    wf.add_roi('aorta')

    wf.dir_manager.add_path('tumor1.nii.gz', 'tumor1', add_to='roi', create=False)
    wf.dir_manager.add_path('pancreas_distal.nii.gz', 'pancreas_distal', add_to='roi', create=False)
    wf.dir_manager.add_path('pancreas_norm.nii.gz', 'pancreas_norm', add_to='roi', create=False)
    for r in ['tumor2','tumor1', 'pancreas_norm', 'pancreas_distal']:
        wf.add_roi(r)
    """
    #wf.crop_volume(wf.dir_manager.get_path('aorta'))

    #wf.calculate_roi_perf()
    #3wf.rois.output()
    #wf.create_perf_map()
    #wf.make_4dvol_frommaps()
    #wf.show_curves()