import matplotlib.pyplot as plt
import subprocess
import sys
from cStringIO import StringIO
import sys
import numpy as np

a=[25.400453333697047,
27.336901904718147,
27.428198302586505,
28.85152528373017,
33.571741470321605,
42.79230861942597,
56.00054983586475,
65.2959243060463,
71.32077683344761,
75.69808055414244,
75.73969608029932,
63.3081467470867,
56.58197878782265,
54.867717451520946,
55.11261037847762,
51.94130780998331,
43.8805877499317,
43.62568170808470,
43.91465937869473]
b=[26.03856606778432,
24.89051614672214,
25.67412558815146,
26.363415759038933,
29.174171922567446,
33.38841688353944,
39.57609222959525,
45.00993386557096,
51.31922257843313,
54.120044403191486,
56.649799993856284,
57.73316217698235,
57.791661003398076,
58.60852199581083,
58.897819223118795,
59.17765218458597,
54.25434011751586,
54.45314695734418,
53.854749441988446]
a=np.array(a)
b=np.array(b)
print a-b

x=10,12,14,16,19,21,23,25,28,30,32,41,46,51,56,61,87,97,107
#plt.subplot(311)
plt.figure(figsize=(7.2, 4.4))
sl=14
plt.plot(x[:sl],a[:sl],'o-',color=(0,1,0), alpha=0.7)
#plt.fill_between(x,a-5,b+5,color='red',where=a>b)
plt.fill_between(x[:sl],a[:sl]-10,a[:sl]+10,alpha=1)
plt.plot(x[:sl],b[:sl],'o-',color='red', alpha=0.7)

#plt.fill_between(x,b-5,b+5,alpha=1,color='green')
#plt.show()

plt.savefig('/home/denest/test1.svg',figsize=(8, 1), dpi=80)
"""
#plt.subplot(312)
plt.plot(x,a,'o-')
plt.errorbar(x, a, yerr=10)
plt.plot(x,b,'o-',color='red')
plt.fill_between(x, 0, 100, where=((a-b)>10), facecolor='red', alpha=0.5)

plt.savefig('/home/denest/test2.svg')

#plt.subplot(313)
plt.clf()
plt.plot(x,a-b,'o-')
#plt.fill_between(x,-10,10,alpha=0.5)
#plt.hlines(10,10,107)
#plt.hlines(-10,10,107)
#plt.fill_between(x, -10, 30, where=((a-b)>10), facecolor='red', alpha=0.5)

plt.savefig('/home/denest/test3.svg')
plt.show()
"""