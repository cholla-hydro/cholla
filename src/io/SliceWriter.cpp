/*!
 * \file
 * Implements the SliceWriter type
 */

#include "SliceWriter.h"

#include <hdf5.h>

#include "../io/FnameTemplate.h"
#include "../io/io.h"
#include "../utils/error_handling.h"

namespace io
{

SliceWriter::SliceWriter(ParameterMap &pmap, const FieldInfo &field_info) {}

void SliceWriter::operator()(Grid3D &G, struct Parameters P, int nfile, const FnameTemplate &fname_template) const
{
#ifdef HDF5
  hid_t file_id;
  herr_t status;

  // create the filename
  std::string filename = fname_template.format_fname(nfile, "_slice");

  // Create a new file
  file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write header (file attributes)
  G.Write_Header_HDF5(file_id);

  // Write slices of all variables to the output file
  G.Write_Slices_HDF5(file_id);

  // Close the file
  status = H5Fclose(file_id);

  #ifdef MPI_CHOLLA
  if (status < 0) {
    printf("Output_Slices: File write failed. ProcID: %d\n", procID);
    chexit(-1);
  }
  #else   // MPI_CHOLLA is not defined
  if (status < 0) {
    printf("Output_Slices: File write failed.\n");
    exit(-1);
  }
  #endif  // MPI_CHOLLA
#else     // HDF5 is not defined
  printf("Output_Slices only defined for hdf5 writes.\n");
#endif    // HDF5
}

}  // namespace io