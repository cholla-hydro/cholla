import h5py as h5
import numpy as np
import sys, os, glob


loud = False
if(len(sys.argv)==2 and sys.argv[1]=="-loud"):
    loud = True
##    


def GetRI(fname,lbox=2):
    d = h5.File(fname, 'r')

    rho = d['density'][...] 
    xHI = d['HI_density'][...]/(rho)

    rbins = np.arange(0,1.001,0.01)
    r = 0.5*(rbins[0:-1]+rbins[1:]);
    vavg = np.zeros_like(r)
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

    vavg /= (1.0e-10+vnum)

    ri = np.interp(0.5,vavg,r)
    return (d.attrs["t"]*1.0e-3,ri)
##    


def Scan(dname,t,err=0.05,lbox=2):

    assert(os.path.isdir(dname))

    fnames = glob.glob(dname+"/*.h5.0")
    fnames.sort()
    if(len(fnames) > 500):
        fnames = fnames[0::10]
    ##

    for fname in fnames:
        d = h5.File(fname, 'r')
        tf = d.attrs["t"]*1.0e-3
        if(tf>0 and abs(np.log10(t)-np.log10(tf))<0.005):
            break
        ##
    else:
        print("No file for time=%g"%t)
        return
    ##

    (t,ri) = GetRI(fname,lbox=lbox)
    
    re = 5.4/6.6*(1-np.exp(-t/122.4))**0.333333
    dr = np.abs(ri/re-1)
    
    if(dr>err or loud):
        print("Error=%g at t=%g"%(dr,t))
        if(dr > err):
            print("FAILED.")
            sys.exit(1)
        ##
    ##
##
  

Scan("OUT",30)
Scan("OUT",500)

print("PASSED.")
