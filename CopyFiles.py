__author__ = 'denis'
import image


folder='/media/RA2/PatientAr/1.2.392.200036.9116.2.5.1.48.1220740667.1315531265.745991/'
out_folder='/media/RA2/ArjEFILM/
sfolder=dict(PatientsName=[0x10,0x10],StudyDate=[0x8,0x20],StudyID=[0x20,0x10], SeriesNumber=[0x20,0x11],ConvolutionKernel=[0x18,0x1210],FilterType=[0x7005,0x100b])

image.dcm_parser(folder,out_folder)
