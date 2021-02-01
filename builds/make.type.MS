#-- build for Magellanic Stream tests

#-- separated output flag so that it can be overriden in target-specific 
#   for make check
OUTPUT    ?=  -DOUTPUT -DHDF5

DFLAGS    += -DCUDA
DFLAGS    += -DMPI_CHOLLA
DFLAGS    += -DBLOCK
DFLAGS    += -DPRECISION=2
DFLAGS    += -DPPMC
DFLAGS    += -DHLLC
DFLAGS    += -DVL
DFLAGS    += $(OUTPUT)
#DFLAGS    += -DDENSITY_FLOOR
#DFLAGS    += -DTEMPERATURE_FLOOR
