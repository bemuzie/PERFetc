import ploting
import image
import os

adress="/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/filtered"
filelist=os.listdir(adress)
adress_out='/media/63A0113C6D30EAE8/_PERF/SZHANIKOV  O.M. 19.01.1947/filtered'
croppart=8
print filelist
for file in filelist[:]:
    if "FC13ORG_5_0.8_1000_crp120_x265y348z198_tips_blf" in file and "png" not in file:
        print file
        img,hdr, mrx=image.loadnii(adress,file)

        ploting.PlotImg(img,mrx,[105,51,62],5,[68,12,29],file)

