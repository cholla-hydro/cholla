import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob

dir = os.getenv('ALTAIR_ROOT')
if(dir == None):
    print('Environment variable ALTAIR_ROOT must be set.')
    sys.exit()
##
sys.path.insert(1,dir+"/python")

import altair.data

TIME_UNIT = 3.15569e10
LENGTH_UNIT = 3.08567758e21
MASS_UNIT = 1.98847e33
DENSITY_UNIT = (MASS_UNIT/(LENGTH_UNIT*LENGTH_UNIT*LENGTH_UNIT))
VELOCITY_UNIT = (LENGTH_UNIT/TIME_UNIT)
NB_UNIT = (MASS_UNIT/(LENGTH_UNIT*LENGTH_UNIT*LENGTH_UNIT))/1.674e-24
ENERGY_UNIT = (DENSITY_UNIT*VELOCITY_UNIT*VELOCITY_UNIT)


fig = plt.figure(figsize=(8,4))
fig.subplots_adjust(left=0.10,right=0.98,bottom=0.15,top=0.98,hspace=0,wspace=0.1)

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)


ax1.set_xlim(0,1.05)
ax1.set_xlabel(r"$r/L_{\rm box}$")
ax1.set_ylim(-5,0.2)
ax1.set_ylabel(r"$lg(x_{\rm HI}), lg(x_{\rm HII})$")

ax2.tick_params(axis="y",which="both",labelleft=False)
ax2.set_xlim(0,1.05)
ax2.set_xlabel(r"$r/L_{\rm box}$")
ax2.set_ylim(-5,0.2)

pathname = os.path.dirname(sys.argv[0])
if(len(pathname) == 0): pathname = "."
img = plt.imread(pathname+"/ref1_030.png")
ax1.imshow(img,extent=[0,1.05,-5,0.8],aspect="auto",alpha=1)
img = plt.imread(pathname+"/ref1_500.png")
ax2.imshow(img,extent=[0,1.05,-5,0.8],aspect="auto",alpha=1)

aa = []


def PlotC1(ax,t,fnames,color="orange",lbox=2,alpha=0,eps=0.01):
        
    for fname in fnames:
        try:
            d = h5.File(fname, 'r')
            tf = d.attrs["t"]*1.0e-3
            if(abs(np.log10(t)-np.log10(tf))<eps):
                print("tf=",tf)
                break
            ##
        except:
            pass
        ##
    else:
        print("No file for time=%g"%t)
        return
    ##

    rho = d['density'][...] 
    xHI = d['HI_density'][...]/(rho)
    xHII = d['HII_density'][...]/(rho)

    rbins = np.arange(0,1.001,0.01)
    r = 0.5*(rbins[0:-1]+rbins[1:]);
    vavg = np.zeros_like(r)
    vrms = np.zeros_like(r)
    vnum = np.zeros_like(r)

    s = rho.shape
    i = np.arange(0,s[0])
    pz, py, px = np.meshgrid(i,i,i,indexing='ij')
    px = (0.5+px)/s[0]
    py = (0.5+py)/s[1]
    pz = (0.5+pz)/s[2]
    r1 = np.sqrt((px-0.5)**2+(py-0.5)**2+(pz-0.5)**2)*lbox
    v1 = xHI
    h0, bins = np.histogram(r1,bins=rbins)
    h1, bins = np.histogram(r1,bins=rbins,weights=v1)
    h2, bins = np.histogram(r1,bins=rbins,weights=v1**2)
    vnum += h0
    vavg += h1
    vrms += h2

    vavg /= (1.0e-10+vnum)
    vrms = np.sqrt(np.abs(vrms/(1.0e-10+vnum)-vavg**2))

    a = ax.plot(r,np.log10(1.0e-30+vavg),color=color,linestyle="-")
    aa.append(a[0])
    a = ax.plot(r,np.log10(1.0e-30+np.abs(1-vavg)),color=color,linestyle=":")
    aa.append(a[0])
    if(alpha > 0): ax.fill_between(r,np.log10(1.0e-10+np.abs(vavg-vrms)),np.log10(1.0e-10+vavg+vrms),color=color,alpha=alpha)
##    

def PlotA1(ax,t,fnames,color,lbox=2,alpha=0,eps=0.001):

    for fname in fnames:
        f = open(fname,"r")
        assert(f)
        for i in range(3):
            s = f.readline()
        ##
        f.close()
        assert(s[:7] == "time = ")
        tf = float(s[7:15])
        if(abs(t-tf) < eps):
            break
        ##
    else:
        print("No file for time=%g"%t)
        return
    ##
        
    d = altair.data.Load(fname[:-9])

    rbins = np.arange(0,1.001,0.01)
    r = 0.5*(rbins[0:-1]+rbins[1:]);
    vavg = np.zeros_like(r)
    vrms = np.zeros_like(r)
    vnum = np.zeros_like(r)
    
    for p in d.patches:
        if(altair.data.IsLeaf(p)):
            lijk = p['lijk']
            s = 0.5**lijk[0]
            x0 = s*lijk[1:]
            i = np.arange(0,d.size[0])
            pz, py, px = np.meshgrid(i,i,i,indexing='ij')
            px = x0[0] + s*(0.5+px)/d.size[0]
            py = x0[1] + s*(0.5+py)/d.size[1]
            pz = x0[2] + s*(0.5+pz)/d.size[2]
            r1 = np.sqrt((px-0.5)**2+(py-0.5)**2+(pz-0.5)**2)*lbox
            v1 = d.Field(p,"HI-density")/d.Field(p,"gas-mass-density")
            h0, bins = np.histogram(r1,bins=rbins)
            h1, bins = np.histogram(r1,bins=rbins,weights=v1)
            h2, bins = np.histogram(r1,bins=rbins,weights=v1**2)
            vnum += h0
            vavg += h1
            vrms += h2
        ##
    ##

    vavg /= (1.0e-10+vnum)
    vrms = np.sqrt(np.abs(vrms/(1.0e-10+vnum)-vavg**2))

    a = ax.plot(r,np.log10(1.0e-30+vavg),color=color,linestyle="-")
    aa.append(a[0])
    a = ax.plot(r,np.log10(1.0e-30+np.abs(1-vavg)),color=color,linestyle=":")
    aa.append(a[0])
    if(alpha > 0): ax.fill_between(r,np.log10(1.0e-10+np.abs(vavg-vrms)),np.log10(1.0e-10+vavg+vrms),color=color,alpha=alpha)
##    

def Plot2(alt,dname,color,lbox=2,eps=0.01):

    assert(os.path.isdir(dname))

    if(alt):
        ff = glob.glob(dname+"/out.*/manifest")
        ff.sort()
        Fun = PlotA1
    else:
        ff = glob.glob(dname+"/*.h5.0")
        ff.sort()
        Fun = PlotC1
    ##
    
    #Plot1(ax1,10,ff,color)
    Fun(ax1,30,ff,color,eps=eps)
    Fun(ax2,500,ff,color,eps=eps)
##


re = 5.4/6.6*(1-np.exp(-30/122.4))**0.333333
ax1.plot([re,re],[-10,10],"k:")
re = 5.4/6.6*(1-np.exp(-500/122.4))**0.333333
ax2.plot([re,re],[-10,10],"k:")
 

Plot2(1,"/data/gnedin/A/TEST/REF.GPU",color="b")
Plot2(0,"OUT.0.01",color="r")
Plot2(0,"OUT",color="orange")
 

plt.show()

