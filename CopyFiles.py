__author__ = 'denis'
import image


folder='/home/denest/Archive/Temp/1'
out_folder='/home/denest/Archive/Temp/'
sfolder=dict(PatientsName=[0x10,0x10],StudyDate=[0x8,0x20],StudyID=[0x20,0x10], SeriesNumber=[0x20,0x11],ConvolutionKernel=[0x18,0x1210],FilterType=[0x7005,0x100b])

image.dcm_parser(folder,out_folder,sfolder)
