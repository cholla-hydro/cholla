/*!
 * \file ct_electric_fields.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains implementation for the CT electric fields code
 *
 */

// STL Includes

// External Includes

// Local Includes
#include "../mhd/ct_electric_fields.h"

namespace mhd
{
   // =========================================================================
   __global__ void Calculate_CT_Electric_Fields(Real const *fluxX,
                                                Real const *fluxY,
                                                Real const *fluxZ,
                                                Real const *dev_conserved,
                                                Real *ctElectricFields,
                                                int const nx,
                                                int const ny,
                                                int const nz,
                                                int const n_cells)
    {
        // get a thread index
        int const threadId = threadIdx.x + blockIdx.x * blockDim.x;
        int xid, yid, zid;
        cuda_utilities::compute3DIndices(threadId, nx, ny, xid, yid, zid);

        // Thread guard to avoid overrun and to skip the first two cells since
        // those ghost cells can't be reconstructed
        if (    xid > 1
            and yid > 1
            and zid > 1
            and xid < nx
            and yid < ny
            and zid < nz)
        {
            // According to the source code of Athena, the following equation
            // relate the magnetic flux to the face centered electric
            // fields/EMF. -cross(V,B)x is the negative of the x-component of V
            // cross B. Note that "X" is the direction the solver is running in
            // this case, not necessarily the true "X".
            //  F_x[(grid_enum::fluxX_magnetic_z)*n_cells] = VxBy - BxVy = -(-cross(V,B))z = -EMF_Z
            //  F_x[(grid_enum::fluxX_magnetic_y)*n_cells] = VxBz - BxVz =  (-cross(V,B))y =  EMF_Y
            //  F_y[(grid_enum::fluxY_magnetic_x)*n_cells] = VxBy - BxVy = -(-cross(V,B))z = -EMF_X
            //  F_y[(grid_enum::fluxY_magnetic_z)*n_cells] = VxBz - BxVz =  (-cross(V,B))y =  EMF_Z
            //  F_z[(grid_enum::fluxZ_magnetic_y)*n_cells] = VxBy - BxVy = -(-cross(V,B))z = -EMF_Y
            //  F_z[(grid_enum::fluxZ_magnetic_x)*n_cells] = VxBz - BxVz =  (-cross(V,B))y =  EMF_X

            // Notes on Implementation Details
            // - The density flux has the same sign as the velocity on the face
            //   and we only care about the sign so we're using the density flux
            //   to perform upwinding checks
            // - All slopes are computed without the factor of two shown in
            //   Stone & Gardiner 2008 eqn. 24. That factor of two is taken care
            //   of in the final assembly of the electric field

            // Variable to get the sign of the velocity at the interface.
            Real signUpwind;

            // Slope and face variables. Format is
            // "<slope/face>_<direction>_<pos/neg>". Slope/Face indicates if the
            // value is a slope or a face centered EMF, direction indicates the
            // direction of the derivative/face and pos/neg indicates if it's
            // the slope on the positive or negative side of the edge field
            // being computed. Note that the direction for the face is parallel
            // to the face and the other direction that is parallel to that face
            // is the direction of the electric field being calculated
            Real slope_x_pos, slope_x_neg,
                 slope_y_pos, slope_y_neg,
                 slope_z_pos, slope_z_neg,
                 face_x_pos,  face_x_neg,
                 face_y_pos,  face_y_neg,
                 face_z_pos,  face_z_neg;
            // ================
            // X electric field
            // ================

            // Y-direction slope on the positive Y side
            signUpwind = fluxZ[cuda_utilities::compute1DIndex(xid, yid, zid-1, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_y_pos = mhd::_internal::_ctSlope(fluxY, dev_conserved, -1, 0, 2, -1, 1, 2, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_y_pos = mhd::_internal::_ctSlope(fluxY, dev_conserved, -1, 0, -1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_y_pos = 0.5 * (mhd::_internal::_ctSlope(fluxY, dev_conserved, -1, 0,  2, -1, 1,  2, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxY, dev_conserved, -1, 0, -1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // Y-direction slope on the negative Y side
            signUpwind = fluxZ[cuda_utilities::compute1DIndex(xid, yid-1, zid-1, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_y_neg = mhd::_internal::_ctSlope(fluxY, dev_conserved, -1, 0, 1, 2, 1, 2, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_y_neg = mhd::_internal::_ctSlope(fluxY, dev_conserved, -1, 0, 1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_y_neg = 0.5 * (mhd::_internal::_ctSlope(fluxY, dev_conserved, -1, 0, 1,  2, 1,  2, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxY, dev_conserved, -1, 0, 1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // Z-direction slope on the positive Z side
            signUpwind = fluxY[cuda_utilities::compute1DIndex(xid, yid-1, zid, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_z_pos = mhd::_internal::_ctSlope(fluxZ, dev_conserved, 1, 0, 1, -1, 1, 2, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_z_pos = mhd::_internal::_ctSlope(fluxZ, dev_conserved, 1, 0, -1, -1, 2, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_z_pos = 0.5 * (mhd::_internal::_ctSlope(fluxZ, dev_conserved, 1, 0,  1, -1, 1,  2, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxZ, dev_conserved, 1, 0, -1, -1, 2, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // Z-direction slope on the negative Z side
            signUpwind = fluxY[cuda_utilities::compute1DIndex(xid, yid-1, zid-1, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_z_neg = mhd::_internal::_ctSlope(fluxZ, dev_conserved, 1, 0, 1, 2, 1, 2, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_z_neg = mhd::_internal::_ctSlope(fluxZ, dev_conserved, 1, 0, 2, -1, -1, 2, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_z_neg = 0.5 * (mhd::_internal::_ctSlope(fluxZ, dev_conserved, 1, 0, 1,  2, 1,  2, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxZ, dev_conserved, 1, 0, 2, -1, -1, 2, xid, yid, zid, nx, ny, n_cells));
            }

            // Load the face centered electric fields  Note the negative signs to
            // convert from magnetic flux to electric field

            face_y_pos = + fluxZ[cuda_utilities::compute1DIndex(xid  , yid  , zid-1, nx, ny) + (grid_enum::fluxZ_magnetic_x)*n_cells];
            face_y_neg = + fluxZ[cuda_utilities::compute1DIndex(xid  , yid-1, zid-1, nx, ny) + (grid_enum::fluxZ_magnetic_x)*n_cells];
            face_z_pos = - fluxY[cuda_utilities::compute1DIndex(xid  , yid-1, zid  , nx, ny) + (grid_enum::fluxY_magnetic_x)*n_cells];
            face_z_neg = - fluxY[cuda_utilities::compute1DIndex(xid  , yid-1, zid-1, nx, ny) + (grid_enum::fluxY_magnetic_x)*n_cells];

            // sum and average face centered electric fields and slopes to get the
            // edge averaged electric field.
            ctElectricFields[threadId + 0*n_cells] = 0.25 * (+ face_y_pos
                                                             + face_y_neg
                                                             + face_z_pos
                                                             + face_z_neg
                                                             + slope_y_pos
                                                             + slope_y_neg
                                                             + slope_z_pos
                                                             + slope_z_neg);

            // ================
            // Y electric field
            // ================

            // X-direction slope on the positive X side
            signUpwind = fluxZ[cuda_utilities::compute1DIndex(xid, yid, zid-1, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_x_pos = mhd::_internal::_ctSlope(fluxX, dev_conserved, 1, 1, 2, -1, 0, 2, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_x_pos = mhd::_internal::_ctSlope(fluxX, dev_conserved, 1, 1, -1, -1, 0, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_x_pos = 0.5 * (mhd::_internal::_ctSlope(fluxX, dev_conserved, 1, 1,  2, -1, 0,  2, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxX, dev_conserved, 1, 1, -1, -1, 0, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // X-direction slope on the negative X side
            signUpwind = fluxZ[cuda_utilities::compute1DIndex(xid-1, yid, zid-1, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_x_neg = mhd::_internal::_ctSlope(fluxX, dev_conserved, 1, 1, 0, 2, 0, 2, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_x_neg = mhd::_internal::_ctSlope(fluxX, dev_conserved, 1, 1, 0, -1, 0, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_x_neg = 0.5 * (mhd::_internal::_ctSlope(fluxX, dev_conserved, 1, 1, 0,  2, 0,  2, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxX, dev_conserved, 1, 1, 0, -1, 0, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // Z-direction slope on the positive Z side
            signUpwind = fluxX[cuda_utilities::compute1DIndex(xid-1, yid, zid, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_z_pos = mhd::_internal::_ctSlope(fluxZ, dev_conserved, -1, 1, 0, -1, 0, 2, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_z_pos = mhd::_internal::_ctSlope(fluxZ, dev_conserved, -1, 1, -1, -1, 2, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_z_pos = 0.5 * (mhd::_internal::_ctSlope(fluxZ, dev_conserved, -1, 1,  0, -1, 0,  2, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxZ, dev_conserved, -1, 1, -1, -1, 2, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // Z-direction slope on the negative Z side
            signUpwind = fluxX[cuda_utilities::compute1DIndex(xid-1, yid, zid-1, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_z_neg = mhd::_internal::_ctSlope(fluxZ, dev_conserved, -1, 1, 0, 2, 0, 2, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_z_neg = mhd::_internal::_ctSlope(fluxZ, dev_conserved, -1, 1, 2, -1, 2, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_z_neg = 0.5 * (mhd::_internal::_ctSlope(fluxZ, dev_conserved, -1, 1, 0,  2, 0,  2, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxZ, dev_conserved, -1, 1, 2, -1, 2, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // Load the face centered electric fields  Note the negative signs to
            // convert from magnetic flux to electric field
            face_x_pos = - fluxZ[cuda_utilities::compute1DIndex(xid  , yid, zid-1, nx, ny) + (grid_enum::fluxZ_magnetic_y)*n_cells];
            face_x_neg = - fluxZ[cuda_utilities::compute1DIndex(xid-1, yid, zid-1, nx, ny) + (grid_enum::fluxZ_magnetic_y)*n_cells];
            face_z_pos = + fluxX[cuda_utilities::compute1DIndex(xid-1, yid, zid  , nx, ny) + (grid_enum::fluxX_magnetic_y)*n_cells];
            face_z_neg = + fluxX[cuda_utilities::compute1DIndex(xid-1, yid, zid-1, nx, ny) + (grid_enum::fluxX_magnetic_y)*n_cells];

            // sum and average face centered electric fields and slopes to get the
            // edge averaged electric field.
            ctElectricFields[threadId + 1*n_cells] = 0.25 * (+ face_x_pos
                                                             + face_x_neg
                                                             + face_z_pos
                                                             + face_z_neg
                                                             + slope_x_pos
                                                             + slope_x_neg
                                                             + slope_z_pos
                                                             + slope_z_neg);

            // ================
            // Z electric field
            // ================

            // Y-direction slope on the positive Y side
            signUpwind = fluxX[cuda_utilities::compute1DIndex(xid-1, yid, zid, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_y_pos = mhd::_internal::_ctSlope(fluxY, dev_conserved, 1, 2, 0, -1, 0, 1, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_y_pos = mhd::_internal::_ctSlope(fluxY, dev_conserved, 1, 2, -1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_y_pos = 0.5 * (mhd::_internal::_ctSlope(fluxY, dev_conserved, 1, 2,  0, -1, 0,  1, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxY, dev_conserved, 1, 2, -1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // Y-direction slope on the negative Y side
            signUpwind = fluxX[cuda_utilities::compute1DIndex(xid-1, yid-1, zid, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_y_neg = mhd::_internal::_ctSlope(fluxY, dev_conserved, 1, 2, 0, 1, 0, 1, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_y_neg = mhd::_internal::_ctSlope(fluxY, dev_conserved, 1, 2, 1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_y_neg = 0.5 * (mhd::_internal::_ctSlope(fluxY, dev_conserved, 1, 2, 0,  1, 0,  1, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxY, dev_conserved, 1, 2, 1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // X-direction slope on the positive X side
            signUpwind = fluxY[cuda_utilities::compute1DIndex(xid, yid-1, zid, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_x_pos = mhd::_internal::_ctSlope(fluxX, dev_conserved, -1, 2, 1, -1, 0, 1, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_x_pos = mhd::_internal::_ctSlope(fluxX, dev_conserved, -1, 2, -1, -1, 0, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_x_pos = 0.5 * (mhd::_internal::_ctSlope(fluxX, dev_conserved, -1, 2,  1, -1, 0,  1, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxX, dev_conserved, -1, 2, -1, -1, 0, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // X-direction slope on the negative X side
            signUpwind = fluxY[cuda_utilities::compute1DIndex(xid-1, yid-1, zid, nx, ny)];
            if (signUpwind > 0.0)
            {
                slope_x_neg = mhd::_internal::_ctSlope(fluxX, dev_conserved, -1, 2, 0, 1, 0, 1, xid, yid, zid, nx, ny, n_cells);
            }
            else if (signUpwind < 0.0)
            {
                slope_x_neg = mhd::_internal::_ctSlope(fluxX, dev_conserved, -1, 2, 0, -1, 0, -1, xid, yid, zid, nx, ny, n_cells);
            }
            else
            {
                slope_x_neg = 0.5 * (mhd::_internal::_ctSlope(fluxX, dev_conserved, -1, 2, 0,  1, 0,  1, xid, yid, zid, nx, ny, n_cells)
                                   + mhd::_internal::_ctSlope(fluxX, dev_conserved, -1, 2, 0, -1, 0, -1, xid, yid, zid, nx, ny, n_cells));
            }

            // Load the face centered electric fields  Note the negative signs to
            // convert from magnetic flux to electric field
            face_x_pos = + fluxY[cuda_utilities::compute1DIndex(xid  , yid-1, zid, nx, ny) + (grid_enum::fluxY_magnetic_z)*n_cells];
            face_x_neg = + fluxY[cuda_utilities::compute1DIndex(xid-1, yid-1, zid, nx, ny) + (grid_enum::fluxY_magnetic_z)*n_cells];
            face_y_pos = - fluxX[cuda_utilities::compute1DIndex(xid-1, yid  , zid, nx, ny) + (grid_enum::fluxX_magnetic_z)*n_cells];
            face_y_neg = - fluxX[cuda_utilities::compute1DIndex(xid-1, yid-1, zid, nx, ny) + (grid_enum::fluxX_magnetic_z)*n_cells];

            // sum and average face centered electric fields and slopes to get the
            // edge averaged electric field.
            ctElectricFields[threadId + 2*n_cells] = 0.25 * (+ face_x_pos
                                                             + face_x_neg
                                                             + face_y_pos
                                                             + face_y_neg
                                                             + slope_x_pos
                                                             + slope_x_neg
                                                             + slope_y_pos
                                                             + slope_y_neg);
        }
    }
    // =========================================================================
} // end namespace mhd