EXEC   = cholla

OPTIMIZE =  -O2  


OBJS   = main.o global.o grid3D.o initial_conditions.o boundary_conditions.o CTU_1D.o CTU_2D.o CTU_3D.o plmp.o plmc.o ppmp.o ppmc.o exact.o roe.o subgrid_routines_2D.o subgrid_routines_3D.o  mpi_routines.o mpi_boundaries.o MPI_Comm_node.o io_mpi.o io.o error_handling.o rng.o global_cuda.o CTU_1D_cuda.o CTU_2D_cuda.o CTU_3D_cuda.o VL_1D_cuda.o VL_2D_cuda.o VL_3D_cuda.o pcm_cuda.o plmp_ctu_cuda.o plmc_ctu_cuda.o plmp_vl_cuda.o ppmp_ctu_cuda.o ppmc_ctu_cuda.o ppmp_vl_cuda.o ppmc_vl_cuda.o exact_cuda.o roe_cuda.o h_correction_2D_cuda.o h_correction_3D_cuda.o cooling.o cuda_mpi_routines.o


#To use MPI, MPI_FLAGS must be set to -DMPI_CHOLLA
#otherwise gcc/g++ will be used for serial compilation
MPI_FLAGS =  -DMPI_CHOLLA

ifdef MPI_FLAGS
  CC	= mpicc
  CXX   = mpicxx

  #MPI_FLAGS += -DSLAB
  #FFT_LIBS  += -lfftw3_mpi

  MPI_FLAGS += -DBLOCK

else
  CC	= /usr/bin/gcc
  CXX = /usr/bin/g++
endif

#define the NVIDIA CUDA compiler
NVCC	= nvcc

.SUFFIXES : .c .cpp .cu .o

#PRECISION = -DPRECISION=1
PRECISION = -DPRECISION=2

#OUTPUT = -DBINARY
OUTPUT = -DHDF5

#RECONSTRUCTION = -DPCM
#RECONSTRUCTION = -DPLMP
#RECONSTRUCTION = -DPLMC
RECONSTRUCTION = -DPPMP
#RECONSTRUCTION = -DPPMC

#SOLVER = -DEXACT
SOLVER = -DROE

INTEGRATOR = -DCTU 
#INTEGRATOR = -DVL

INCL   = -I./ -I/usr/local/cuda/include -I/usr/local/include
NVLIBS = -L/usr/local/cuda/lib -lcuda -lcudart
LIBS   = -L/usr/local/lib -lm -lgsl -lhdf5


FLAGS = $(PRECISION) $(OUTPUT) $(RECONSTRUCTION) $(SOLVER) $(INTEGRATOR) -DCUDA -DCUDA_ERROR_CHECK
CFLAGS 	  = $(OPTIMIZE) $(FLAGS) $(MPI_FLAGS) -m64
CXXFLAGS  = $(OPTIMIZE) $(FLAGS) $(MPI_FLAGS) -m64
NVCCFLAGS = $(FLAGS) -m64 -arch=sm_30 -fmad=false -ccbin clang++ -Xcompiler -arch -Xcompiler x86_64
LDFLAGS	  = -m64 -F/Library/Frameworks -framework CUDA


%.o:	%.c
		$(CC) $(CFLAGS)  $(INCL)  -c $< -o $@ 

%.o:	%.cpp
		$(CXX) $(CXXFLAGS)  $(INCL)  -c $< -o $@ 

%.o:	%.cu
		$(NVCC) $(NVCCFLAGS)  $(INCL)  -c $< -o $@ 

$(EXEC): $(OBJS) 
	 	 $(CXX) $(LDFLAGS) $(OBJS) $(LIBS) $(NVLIBS) -o $(EXEC) $(INCL) 

#$(OBJS): $(INCL) 

.PHONY : clean

clean:
	 rm -f $(OBJS) $(EXEC)

