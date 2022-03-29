#-- build for Magellanic Stream tests

#-- separated output flag so that it can be overriden in target-specific 
#   for make check
OUTPUT    ?=  -DOUTPUT -DHDF5 -DSLICES -DPROJECTION

MPI_GPU   ?=

DFLAGS    += -DCUDA
#DFLAGS    += -DMPI_CHOLLA 
DFLAGS    += -DPRECISION=2
DFLAGS    += -DPPMP
DFLAGS    += -DHLLC
DFLAGS    += -DVL

# Apply a density and temperature floor
#DFLAGS    += -DDENSITY_FLOOR
#DFLAGS    += -DTEMPERATURE_FLOOR

# Solve the Gas Internal Energy usisng a Dual Energy Formalism
DFLAGS    += -DDE

# Apply the cooling in the GPU from precomputed tables
DFLAGS    += -DCOOLING_GPU

# Evolve a passive scalar field
DFLAGS    += -DSCALAR

#Measure the Timing of the different stages
#DFLAGS    += -DCPU_TIME

DFLAGS    += $(OUTPUT)

#Select if the Hydro Conserved data will reside in the GPU
#and the MPI transfers are done from the GPU
#If not specified, MPI_GPU is off by default 
#This is set in the system make.host file
DFLAGS    += $(MPI_GPU)  
