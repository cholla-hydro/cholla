import numpy as np
import matplotlib.pyplot as plt
import sys, os


fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0.15,right=0.98,bottom=0.12,top=0.98,hspace=0,wspace=0.08)

axx = fig.subplots(2,1,sharex='col')


axx[1].set_xlim(-6,7.5)
axx[1].set_xlabel(r"$lg(t) [{\rm yr}]$")

axx[0].set_ylim(-7.5,0.2)
axx[0].set_ylabel(r"$lg(x_{\rm HI})$")
axx[1].set_ylim(2,5)
axx[1].set_ylabel(r"$lg(T) [{\rm K}]$")


pathname = os.path.dirname(sys.argv[0])
img0 = plt.imread(pathname+"/ref0.png")
axx[0].imshow(img0,extent=[axx[0].get_xlim()[0],axx[0].get_xlim()[1],axx[0].get_ylim()[0],axx[0].get_ylim()[1]],aspect="auto")

img1 = plt.imread(pathname+"/ref1.png")
axx[1].imshow(img1,extent=[axx[1].get_xlim()[0],axx[1].get_xlim()[1],axx[1].get_ylim()[0],axx[1].get_ylim()[1]],aspect="auto")


def Plot(fname,color,lw=2):
    d = np.loadtxt(fname,unpack=True,skiprows=1)
    d = np.log10(d)

    axx[0].plot(d[0],d[1],linewidth=lw,color=color)
    axx[1].plot(d[0],d[2],linewidth=lw,color=color)


Plot("cholla.res","b")


plt.show()

