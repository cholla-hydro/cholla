import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob

dir = os.getenv('ALTAIR_ROOT')
if(dir == None):
    withALTAIR = False
else:
    withALTAIR = True
    sys.path.insert(1,dir+"/python")
    import altair.data
##


TIME_UNIT = 3.15569e10
LENGTH_UNIT = 3.08567758e21
MASS_UNIT = 1.98847e33
DENSITY_UNIT = (MASS_UNIT/(LENGTH_UNIT*LENGTH_UNIT*LENGTH_UNIT))
VELOCITY_UNIT = (LENGTH_UNIT/TIME_UNIT)
NB_UNIT = (MASS_UNIT/(LENGTH_UNIT*LENGTH_UNIT*LENGTH_UNIT))/1.674e-24
ENERGY_UNIT = (DENSITY_UNIT*VELOCITY_UNIT*VELOCITY_UNIT)


fig = plt.figure(figsize=(9,9))
fig.subplots_adjust(left=0.10,right=0.98,bottom=0.07,top=0.98,hspace=0.05,wspace=0.08)

axx = fig.subplots(3,3,sharex='all',sharey='row')


axx[0][0].set_xlim(0,1.05)
axx[2][0].set_xlabel(r"$r/L_{\rm box}$")
axx[2][1].set_xlabel(r"$r/L_{\rm box}$")
axx[2][2].set_xlabel(r"$r/L_{\rm box}$")

axx[0][0].set_ylim(-5,0.8)
axx[0][0].set_ylabel(r"$lg(x_{\rm HI}), lg(x_{\rm HII})$")
axx[1][0].set_ylim(3.5,4.6)
axx[1][0].set_ylabel(r"$lg(T) [{\rm K}]$")
axx[2][0].set_ylim(-4,0.8)
axx[2][0].set_ylabel(r"$lg(n) [{\rm cm}^{-3}]$")

#ax2.tick_params(axis="y",which="both",labelleft=False)


if(dir):
    pathname = dir + "/tests/rt/iliev6"
    img = plt.imread(pathname+"/ref3x.png")
    axx[0][0].imshow(img,extent=[0,1.05,-5,0.8],aspect="auto",alpha=1)
    img = plt.imread(pathname+"/ref10x.png")
    axx[0][1].imshow(img,extent=[0,1.05,-5,0.8],aspect="auto",alpha=1)
    img = plt.imread(pathname+"/ref25x.png")
    axx[0][2].imshow(img,extent=[0,1.05,-5,0.8],aspect="auto",alpha=1)
    img = plt.imread(pathname+"/ref3T.png")
    axx[1][0].imshow(img,extent=[0,1.05,3.5,4.6],aspect="auto",alpha=1)
    img = plt.imread(pathname+"/ref10T.png")
    axx[1][1].imshow(img,extent=[0,1.05,3.5,4.6],aspect="auto",alpha=1)
    img = plt.imread(pathname+"/ref25T.png")
    axx[1][2].imshow(img,extent=[0,1.05,3.5,4.6],aspect="auto",alpha=1)
    img = plt.imread(pathname+"/ref3n.png")
    axx[2][0].imshow(img,extent=[0,1.05,-4,0.8],aspect="auto",alpha=1)
    img = plt.imread(pathname+"/ref10n.png")
    axx[2][1].imshow(img,extent=[0,1.05,-4,0.8],aspect="auto",alpha=1)
    img = plt.imread(pathname+"/ref25n.png")
    axx[2][2].imshow(img,extent=[0,1.05,-4,0.8],aspect="auto",alpha=1)
##

def PlotC1(axx,t,fnames,color="orange",lbox=2,alpha=0,eps=0.01,lw=2,labs=False):
        
    for fname in fnames:
        try:
            d = h5.File(fname, 'r')
            tf = d.attrs["t"]*1.0e-3
            if(tf>0 and abs(np.log10(t)-np.log10(tf))<eps):
                print("fname=",fname," tf=",tf)
                break
            ##
        except:
            pass
        ##
    else:
        print("Cholla: no file for time=%g"%t)
        return
    ##

    rho = d['density'][...] 
    vx = d['momentum_x'][...]/rho
    vy = d['momentum_x'][...]/rho
    vz = d['momentum_x'][...]/rho
    xHI = d['HI_density'][...]/(rho)
    xHII = d['HII_density'][...]/(rho)
    nb = rho*NB_UNIT

    U = ENERGY_UNIT*(d['Energy'][...]-0.5*rho*(vx**2+vy**2+vz**2))
    T = U/(1.5*nb*1.38e-16*(1+xHII))
    print(T.min(),T.max())
    
    rbins = np.arange(0,1.001,0.01)
    r = 0.5*(rbins[0:-1]+rbins[1:]);
    vavg = np.zeros_like(r)
    vrms = np.zeros_like(r)
    vnum = np.zeros_like(r)
    tavg = np.zeros_like(r)
    navg = np.zeros_like(r)

    s = rho.shape
    i = np.arange(0,s[0])
    pz, py, px = np.meshgrid(i,i,i,indexing='ij')
    px = (0.5+px)/s[0]
    py = (0.5+py)/s[1]
    pz = (0.5+pz)/s[2]
    r1 = np.sqrt((px-0.5)**2+(py-0.5)**2+(pz-0.5)**2)*lbox
    v1 = xHI
    t1 = T
    n1 = nb
    h0, bins = np.histogram(r1,bins=rbins)
    h1, bins = np.histogram(r1,bins=rbins,weights=v1)
    h2, bins = np.histogram(r1,bins=rbins,weights=v1**2)
    vnum += h0
    vavg += h1
    vrms += h2
    h1, bins = np.histogram(r1,bins=rbins,weights=t1)
    tavg += h1
    h1, bins = np.histogram(r1,bins=rbins,weights=n1)
    navg += h1
    vavg /= (1.0e-10+vnum)
    vrms = np.sqrt(np.abs(vrms/(1.0e-10+vnum)-vavg**2))
    tavg /= (1.0e-10+vnum)

    navg /= (1.0e-10+vnum)

    axx[0].plot(r,np.log10(1.0e-30+vavg),color=color,linestyle="-")
    axx[0].plot(r,np.log10(1.0e-30+np.abs(1-vavg)),color=color,linestyle=":")
    if(alpha > 0): axx[0].fill_between(r,np.log10(1.0e-10+np.abs(vavg-vrms)),np.log10(1.0e-10+vavg+vrms),color=color,alpha=alpha)
    axx[1].plot(r,np.log10(1.0e-30+tavg),color=color,linestyle="-")
    axx[2].plot(r,np.log10(1.0e-30+navg),color=color,linestyle="-")
##


def PlotA1(axx,t,fnames,color,lbox=2,alpha=0):

    assert(withALTAIR)
    
    for fname in fnames:
        f = open(fname,"r")
        assert(f)
        for i in range(3):
            s = f.readline()
        ##
        f.close()
        assert(s[:7] == "time = ")
        tf = float(s[7:15])
        if(abs(t-tf)<0.001):
            break
        ##
    else:
        print("No file for time=%g"%t)
        return
    ##
        
    d = altair.data.Load(fname[:-9])

    rbins = np.arange(0.01,1.001,0.01)
    r = 0.5*(rbins[0:-1]+rbins[1:]);
    vavg = np.zeros_like(r)
    vrms = np.zeros_like(r)
    vnum = np.zeros_like(r)
    tavg = np.zeros_like(r)
    navg = np.zeros_like(r)
    
    for p in d.patches:
        if(altair.data.IsLeaf(p)):
            lijk = p['lijk']
            s = 0.5**lijk[0]
            x0 = s*lijk[1:]
            i = np.arange(0,d.size[0])
            if(d.dim == 3):
                pz, py, px = np.meshgrid(i,i,i,indexing='ij')
                px = x0[0] + s*(0.5+px)/d.size[0]
                py = x0[1] + s*(0.5+py)/d.size[1]
                pz = x0[2] + s*(0.5+pz)/d.size[2]
                r1 = np.sqrt((px-0.5)**2+(py-0.5)**2+(pz-0.5)**2)*lbox
            else:
                py, px = np.meshgrid(i,i,indexing='ij')
                px = x0[0] + s*(0.5+px)/d.size[0]
                py = x0[1] + s*(0.5+py)/d.size[1]
                r1 = np.sqrt((px-0.5)**2+(py-0.5)**2)*lbox
            ##
            v1 = d.Field(p,"HI-density")/d.Field(p,"gas-mass-density")
            n1 = d.Field(p,"den")/1.67067e-24
            t1 = d.Field(p,"tem")
            h0, bins = np.histogram(r1,bins=rbins)
            h1, bins = np.histogram(r1,bins=rbins,weights=v1)
            h2, bins = np.histogram(r1,bins=rbins,weights=v1**2)
            vnum += h0
            vavg += h1
            vrms += h2
            h1, bins = np.histogram(r1,bins=rbins,weights=t1)
            tavg += h1
            h1, bins = np.histogram(r1,bins=rbins,weights=n1)
            navg += h1
        ##
    ##

    vavg /= (1.0e-10+vnum)
    vrms = np.sqrt(np.abs(vrms/(1.0e-10+vnum)-vavg**2))
    tavg /= (1.0e-10+vnum)
    navg /= (1.0e-10+vnum)

    axx[0].plot(r,np.log10(1.0e-30+vavg),color=color,linestyle="-")
    axx[0].plot(r,np.log10(1.0e-30+np.abs(1-vavg)),color=color,linestyle=":")
    if(alpha > 0): axx[0].fill_between(r,np.log10(1.0e-10+np.abs(vavg-vrms)),np.log10(1.0e-10+vavg+vrms),color=color,alpha=alpha)
    axx[1].plot(r,np.log10(1.0e-30+tavg),color=color,linestyle="-")
    axx[2].plot(r,np.log10(1.0e-30+navg),color=color,linestyle="-")
##    


def Plot2(dname,color,lbox=2,times=[3,10,25]):

    assert(os.path.isdir(dname))

    ff = glob.glob(dname+"/out.*/manifest")
    if(ff):
        assert(withALTAIR)
        ff.sort()
        Fun = PlotA1
    else:
        ff = glob.glob(dname+"/*.h5.0")
        ff.sort()
        Fun = PlotC1
    ##

    Fun([axx[0][0],axx[1][0],axx[2][0]],times[0],ff,color)
    Fun([axx[0][1],axx[1][1],axx[2][1]],times[1],ff,color)
    Fun([axx[0][2],axx[1][2],axx[2][2]],times[2],ff,color)
##


Plot2("OUT",color="orange")


plt.show()

