/*!
 * \file
 * Implements the RotatedProjWriter type
 */

#include "RotatedProjWriter.h"

#include "io.h"

void io::RotatedProjWriter::operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template)
{
  Output_Rotated_Projected_Data(G, P, nfile, fname_template);
}