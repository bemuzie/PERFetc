__author__ = 'denis'
import image


folder='/media/63A0113C6D30EAE8/_PERF/temp/270112_20120203_102412074'
out_folder='/media/63A0113C6D30EAE8/_PERF'
sfolder=dict(PatientsName=[0x10,0x10],StudyDate=[0x8,0x20],StudyID=[0x20,0x10],\
    SeriesNumber=[0x20,0x11],ConvolutionKernel=[0x18,0x1210],FilterType=[0x7005,0x100b])

image.dcm_parser(folder,out_folder,sfolder)