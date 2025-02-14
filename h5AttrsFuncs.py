# update header flags of old .h5 files

import h5py
import numpy as np

def readFilePrintHeaders(fname):
    
    # function to read in .h5 file,
    # return a list of all attribute keys,
    # and add 'cholla' flag if none exists
    
    file = h5py.File(str(fname), 'r+')
    if 'cholla' in file.attrs.keys():
        return file.attrs.keys()
    else:
        file.attrs.create('cholla', [''])
        return file.attrs.keys()
    
def modifyAttrsVal(fname, Attrs, vals):

    # function to modify one (or more) attributes
    # to provided values for a given .h5 file

    Attrs=np.array(Attrs) # converts input attrs and vals to numpy arrays
    vals=np.array(vals)   # to avoid errors with enumerate

    file = h5py.File(str(fname), 'r+')
    for i, attr in enumerate(Attrs):
        file.attrs.modify(attr, vals[i])
        return f'{attr} updated to {vals[i]}'