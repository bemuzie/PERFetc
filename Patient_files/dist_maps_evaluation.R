library(oro.nifti)
library(ggplot2)
library(tidyr)
library(dplyr)
dist_path='/home/denest/Share/BELODED  I.E. 22.07.1940/20131224_1841/ROI/tumor_roi2_dist_map.nii'
nii_folder='/home/denest/Share/BELODED  I.E. 22.07.1940/20131224_1841/NII/FILTERED'
perf_map_path='/home/denest/Share/BELODED  I.E. 22.07.1940/20131224_1841/NII/RAW/MAPS_CROPED'

dist <- readNIfTI(dist_path)
df<-lapply(paste(perf_map_path,'/',list.files(perf_map_path),sep=''),
       function(x) readNIfTI(x)[dist>0])
df = data.frame(dist=dist[dist>0],df)
names(df) <- c('dist',1:3)

df <- gather(df,ser,value,-dist)
names(df)
df %>%
  filter(ser==1,value>0)%>%
  group_by(dist)%>%
  summarise(mean_val=mean(value))->df_sum
  ggplot(aes(x=dist,y=value))+
    geom_point(alpha=0.1)+
    stat_smooth(method = "gam", formula = y ~ s(x))+
    facet_grid(ser~.)
