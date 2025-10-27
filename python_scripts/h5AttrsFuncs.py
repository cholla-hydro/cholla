# update header flags of old .h5 files

import h5py
import numpy as np


def readFilePrintHeaders(fname):
    # function to read in .h5 file,
    # return a list of all attribute keys,
    # and add 'cholla' flag if none exists

    file = h5py.File(str(fname), "r+")
    if "cholla" in file.attrs.keys():
        return file.attrs.keys()
    else:
        file.attrs.create("cholla", [""])
        return file.attrs.keys()


def modifyAttrsVal(fname, Attrs, vals):
    # function to modify one (or more) attributes
    # to provided values for a given .h5 file

    Attrs = np.array(Attrs)  # converts input attrs and vals to numpy arrays
    vals = np.array(vals)  # to avoid errors with enumerate

    file = h5py.File(str(fname), "r+")
    if Attrs.ndim == 0:
        file.attrs.create(str(Attrs), vals)
    else:
        for i, atts in enumerate(Attrs):
            file.attrs.create(Attrs[i], vals[i])


if __name__ == "__main__":
    fname = str(input("file name = "))
    Attrs = [str(x) for x in input("attributes = ").split()]
    if str(Attrs[0]) == "None":
        chollaFlag = readFilePrintHeaders(fname)
        print(chollaFlag)
    elif Attrs != "None":
        Attrs = np.array(Attrs)
        if Attrs.ndim == 0:
            vals = [int(x) for x in input("values = ").split()]
            mod = modifyAttrsVal(fname, Attrs, vals)
        else:
            valMult = []
            for i in np.array(Attrs):
                vals = [int(x) for x in input("values = ").split()]
                valMult.append(vals)
            mod = modifyAttrsVal(fname, Attrs, valMult)
