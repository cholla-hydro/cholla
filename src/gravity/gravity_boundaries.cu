#ifdef GRAVITY

  #include <cmath>
  #include <utility>

  #include "../gravity/grav3D.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../model/disk_galaxy.h"
  #include "../model/potentials.h"
  #include "../utils/error_handling.h"

/*! \brief aggregates properties of the buffer for boundary vals of the potential */
struct BoundaryBufProps {
  int n_i;
  int n_j;
  int nGHST;
};

/*! \brief retrieve the boundary_buf for the potential and associated properties */
[[maybe_unused]] static std::pair<Real *, BoundaryBufProps> Get_Boundary_Buf_(const Grav3D &Grav, int direction,
                                                                              int side)
{
  // maybe we should convert direction and side to enums (so its more obvious)
  CHOLLA_ASSERT(0 <= direction and direction <= 2, "sanity check failed");
  CHOLLA_ASSERT(side == 0 or side == 1, "sanity check failed");

  int n_i, n_j;
  Real *pot_boundary = nullptr;
  if (direction == 0) {
    n_i = Grav.ny_local;
    n_j = Grav.nz_local;
  #ifdef GRAV_ISOLATED_BOUNDARY_X
    pot_boundary = (side == 0) ? Grav.F.pot_boundary_x0 : Grav.F.pot_boundary_x1;
  #endif
  } else if (direction == 1) {
    n_i = Grav.nx_local;
    n_j = Grav.nz_local;
  #ifdef GRAV_ISOLATED_BOUNDARY_Y
    pot_boundary = (side == 0) ? Grav.F.pot_boundary_y0 : Grav.F.pot_boundary_y1;
  #endif
  } else {  // direction == 2
    n_i = Grav.nx_local;
    n_j = Grav.ny_local;
  #ifdef GRAV_ISOLATED_BOUNDARY_Z
    pot_boundary = (side == 0) ? Grav.F.pot_boundary_z0 : Grav.F.pot_boundary_z1;
  #endif
  }

  CHOLLA_ASSERT(pot_boundary != nullptr, "sanity check failed!");
  return {pot_boundary, {n_i, n_j, N_GHOST_POTENTIAL}};
}

  #if defined(GRAV_ISOLATED_BOUNDARY_X) || defined(GRAV_ISOLATED_BOUNDARY_Y) || defined(GRAV_ISOLATED_BOUNDARY_Z)
void Grid3D::Compute_Potential_Boundaries_Isolated(int dir, struct Parameters *P)
{
  // Set Isolated Boundaries for the ghost cells.
  int bc_potential_type = P->bc_potential_type;
  CHOLLA_ASSERT(bc_potential_type >= 0,
                "bc_potential_type must be set to a non-negative value when simulating "
                "non-periodic boundaries");
  // bc_potential_type = 0 -> Point mass potential GM/r
  if (dir == 0) {
    Compute_Potential_Isolated_Boundary(0, 0, bc_potential_type);
  }
  if (dir == 1) {
    Compute_Potential_Isolated_Boundary(0, 1, bc_potential_type);
  }
  if (dir == 2) {
    Compute_Potential_Isolated_Boundary(1, 0, bc_potential_type);
  }
  if (dir == 3) {
    Compute_Potential_Isolated_Boundary(1, 1, bc_potential_type);
  }
  if (dir == 4) {
    Compute_Potential_Isolated_Boundary(2, 0, bc_potential_type);
  }
  if (dir == 5) {
    Compute_Potential_Isolated_Boundary(2, 1, bc_potential_type);
  }
}

void Grid3D::Set_Potential_Boundaries_Isolated(int direction, int side, int *flags)
{
  std::pair<Real *, BoundaryBufProps> tmp = Get_Boundary_Buf_(Grav, direction, side);
  Real *pot_boundary                      = tmp.first;
  int n_i                                 = tmp.second.n_i;
  int n_j                                 = tmp.second.n_j;
  int nGHST                               = tmp.second.nGHST;

  int nx_g, ny_g, nz_g;
  int nx_local, ny_local, nz_local;
  nx_g     = Grav.nx_local + 2 * nGHST;
  ny_g     = Grav.ny_local + 2 * nGHST;
  nz_g     = Grav.nz_local + 2 * nGHST;
  nx_local = Grav.nx_local;
  ny_local = Grav.ny_local;
  nz_local = Grav.nz_local;

  int i, j, k, id_buffer, id_grid;

  for (k = 0; k < nGHST; k++) {
    for (i = 0; i < n_i; i++) {
      for (j = 0; j < n_j; j++) {
        id_buffer = i + j * n_i + k * n_i * n_j;

        if (direction == 0) {
          if (side == 0) {
            id_grid = (k) + (i + nGHST) * nx_g + (j + nGHST) * nx_g * ny_g;
          }
          if (side == 1) {
            id_grid = (k + nx_local + nGHST) + (i + nGHST) * nx_g + (j + nGHST) * nx_g * ny_g;
          }
        }
        if (direction == 1) {
          if (side == 0) {
            id_grid = (i + nGHST) + (k)*nx_g + (j + nGHST) * nx_g * ny_g;
          }
          if (side == 1) {
            id_grid = (i + nGHST) + (k + ny_local + nGHST) * nx_g + (j + nGHST) * nx_g * ny_g;
          }
        }
        if (direction == 2) {
          if (side == 0) {
            id_grid = (i + nGHST) + (j + nGHST) * nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            id_grid = (i + nGHST) + (j + nGHST) * nx_g + (k + nz_local + nGHST) * nx_g * ny_g;
          }
        }

        Grav.F.potential_h[id_grid] = pot_boundary[id_buffer];
      }
    }
  }
}

/*! \brief A helper function that computes the gravitational potential at the
 *         simulation boundaries using a callback function
 *
 *  \param[out] pot_boundary the buffer to be filled with the computed potential
 *  \param[in] boundary_buf_props Describes properties of \p pot_boundary
 *  \param[in] Grav Holds information needed to compute the spatial location of each
 *      location.
 *  \param[in] direction Encodes the axis of the boundary
 *  \param[in] side Encodes whether we are consider a left or right boundary
 *  \param[in] fn A function object that effectively has the signature
 *      `Real fn(Real x, Real y, Real z)`. In more detail, it returns the potential
 *      computed at a specified position.
 *
 *  \note
 *  Ideally, we might replace \p Grav with an instance of \ref SpatialDomainProps (or
 *  something similar). This will become in a future planned change that will factor
 *  out this logic and other logic pertaining the estimate for the dynamical potential.
 *  It's also more elegant (i.e. \ref Grav3D contains a lot of superfluous info) and
 *  makes it possible to invoke this logic on GPUs (we would just need to replace the
 *  for-loop with \ref gpuFor)
 */
template <typename PotentialFn>
static void Compute_Potential_Isolated_Boundary_Helper(Real *pot_boundary, const BoundaryBufProps &boundary_buf_props,
                                                       const Grav3D &Grav, int direction, int side, PotentialFn fn)
{
  Real Lx_local = Grav.nx_local * Grav.dx;
  Real Ly_local = Grav.ny_local * Grav.dy;
  Real Lz_local = Grav.nz_local * Grav.dz;

  int n_i   = boundary_buf_props.n_i;
  int n_j   = boundary_buf_props.n_j;
  int nGHST = boundary_buf_props.nGHST;

  for (int k = 0; k < nGHST; k++) {
    for (int i = 0; i < n_i; i++) {
      for (int j = 0; j < n_j; j++) {
        int id = i + j * n_i + k * n_i * n_j;

        // calculate the position
        Real pos_x, pos_y, pos_z;
        if (direction == 0) {
          // pos_x = Grav.xMin - ( nGHST + k + 0.5 ) * Grav.dx;
          pos_x = Grav.xMin + (k + 0.5 - nGHST) * Grav.dx;
          if (side == 1) {
            pos_x += Lx_local + nGHST * Grav.dx;
          }
          pos_y = Grav.yMin + (i + 0.5) * Grav.dy;
          pos_z = Grav.zMin + (j + 0.5) * Grav.dz;
        } else if (direction == 1) {
          // pos_y = Grav.yMin - ( nGHST + k + 0.5 ) * Grav.dy;
          pos_y = Grav.yMin + (k + 0.5 - nGHST) * Grav.dy;
          if (side == 1) {
            pos_y += Ly_local + nGHST * Grav.dy;
          }
          pos_x = Grav.xMin + (i + 0.5) * Grav.dx;
          pos_z = Grav.zMin + (j + 0.5) * Grav.dz;
        } else {  // (direction == 2)
          // pos_z = Grav.zMin - ( nGHST + k + 0.5 ) * Grav.dz;
          pos_z = Grav.zMin + (k + 0.5 - nGHST) * Grav.dz;
          if (side == 1) {
            pos_z += Lz_local + nGHST * Grav.dz;
          }
          pos_x = Grav.xMin + (i + 0.5) * Grav.dx;
          pos_y = Grav.yMin + (j + 0.5) * Grav.dy;
        }
        pot_boundary[id] = fn(pos_x, pos_y, pos_z);
      }
    }
  }
}

void Grid3D::Compute_Potential_Isolated_Boundary(int direction, int side, int bc_potential_type)
{
  std::pair<Real *, BoundaryBufProps> tmp = Get_Boundary_Buf_(Grav, direction, side);
  auto [pot_boundary, boundary_buf_props] = tmp;

  if (bc_potential_type == 0) {
    // Point mass potential GM/r
    const Real r0       = H.sphere_radius;
    const Real M        = (H.sphere_density - H.sphere_background_density) * 4.0 * M_PI * r0 * r0 * r0 / 3.0;
    const Real cm_pos_x = H.sphere_center_x;
    const Real cm_pos_y = H.sphere_center_y;
    const Real cm_pos_z = H.sphere_center_z;

    // define a local function that actually computes the potential
    auto calc_potential = [=](Real pos_x, Real pos_y, Real pos_z) -> Real {
      Real delta_x = pos_x - cm_pos_x;
      Real delta_y = pos_y - cm_pos_y;
      Real delta_z = pos_z - cm_pos_z;
      Real r       = sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z));
      return -Grav.Gconst * M / r;
    };

    // now, use the calc_potential function to actually fill the boundaries
    Compute_Potential_Isolated_Boundary_Helper(pot_boundary, boundary_buf_props, Grav, direction, side, calc_potential);
  } else if (bc_potential_type == 1) {
    // M-W disk potential

    // The underlying assumption of PARIS_GALACTIC is that we have a good analytic
    // approximation the gravitation potential at the boundaries due to the dynamical density
    // (i.e. gas density and particle density)
    // - we implicitly make use of that potential when solving for the potential
    // - we also make use of it here to overwrite the values of the potential at the boundary

    // Currently, we need to make sure this stays synchronized with the approximation used within
    // Paris_Galactic. We should refactor so that we don't need to do that
    // Right now:
    // -> we are implicitly assuming that the gas disk is the only source of dynamical density
    //    (i.e. the `rho_real` array is dominated by gas density)
    // -> we are currently ignoring contributions from particles
    const ApproxExponentialDisk3MN approx_potential = galaxies::MW.getGasDisk().selfgrav_approx_potential;

    // define a local function that actually computes the potential
    auto calc_potential = [=](Real pos_x, Real pos_y, Real pos_z) -> Real {
      Real r = sqrt((pos_x * pos_x) + (pos_y * pos_y));
      return approx_potential.phi_disk_D3D(r, pos_z);
    };

    // now, use the calc_potential function to actually fill the boundaries
    Compute_Potential_Isolated_Boundary_Helper(pot_boundary, boundary_buf_props, Grav, direction, side, calc_potential);
  } else {
    CHOLLA_ERROR("Invalid bc_potential_type value: %d", bc_potential_type);
  }
}

  #endif  // GRAV_ISOLATED_BOUNDARY_X

void Grid3D::Set_Potential_Boundaries_Periodic(int direction, int side, int *flags)
{
  // Flags: 1 (periodic), 2 (reflective), 3 (transmissive), 4 (custom), 5 (mpi)

  int i, j, k, indx_src, indx_dst;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = N_GHOST_POTENTIAL;
  nx_g  = Grav.nx_local + 2 * nGHST;
  ny_g  = Grav.ny_local + 2 * nGHST;
  nz_g  = Grav.nz_local + 2 * nGHST;

  // Copy X boundaries
  if (direction == 0) {
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nGHST; i++) {
          if (side == 0) {
            indx_src = (nx_g - 2 * nGHST + i) + (j)*nx_g + (k)*nx_g * ny_g;  // Periodic
            indx_dst = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx_src = (i + nGHST) + (j)*nx_g + (k)*nx_g * ny_g;  // Periodic
            indx_dst = (nx_g - nGHST + i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          Grav.F.potential_h[indx_dst] = Grav.F.potential_h[indx_src];
        }
      }
    }
  }

  // Copy Y boundaries
  if (direction == 1) {
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < nGHST; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx_src = (i) + (ny_g - 2 * nGHST + j) * nx_g + (k)*nx_g * ny_g;  // Periodic
            indx_dst = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx_src = (i) + (j + nGHST) * nx_g + (k)*nx_g * ny_g;  // Periodic
            indx_dst = (i) + (ny_g - nGHST + j) * nx_g + (k)*nx_g * ny_g;
          }
          Grav.F.potential_h[indx_dst] = Grav.F.potential_h[indx_src];
        }
      }
    }
  }

  // Copy Z boundaries
  if (direction == 2) {
    for (k = 0; k < nGHST; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx_src = (i) + (j)*nx_g + (nz_g - 2 * nGHST + k) * nx_g * ny_g;  // Periodic
            indx_dst = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx_src = (i) + (j)*nx_g + (k + nGHST) * nx_g * ny_g;  // Periodic
            indx_dst = (i) + (j)*nx_g + (nz_g - nGHST + k) * nx_g * ny_g;
          }
          Grav.F.potential_h[indx_dst] = Grav.F.potential_h[indx_src];
        }
      }
    }
  }
}

  #ifdef MPI_CHOLLA
int Grid3D::Load_Gravity_Potential_To_Buffer(int direction, int side, Real *buffer, int buffer_start)
{
  int i, j, k, indx, indx_buff, length;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = N_GHOST_POTENTIAL;
  nx_g  = Grav.nx_local + 2 * nGHST;
  ny_g  = Grav.ny_local + 2 * nGHST;
  nz_g  = Grav.nz_local + 2 * nGHST;

  // Load X boundaries
  if (direction == 0) {
    length = nGHST * nz_g * ny_g;
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nGHST; i++) {
          if (side == 0) {
            indx = (i + nGHST) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx = (nx_g - 2 * nGHST + i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          indx_buff                        = (j) + (k)*ny_g + i * ny_g * nz_g;
          buffer[buffer_start + indx_buff] = Grav.F.potential_h[indx];
        }
      }
    }
  }

  // Load Y boundaries
  if (direction == 1) {
    length = nGHST * nz_g * nx_g;
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < nGHST; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx = (i) + (j + nGHST) * nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx = (i) + (ny_g - 2 * nGHST + j) * nx_g + (k)*nx_g * ny_g;
          }
          indx_buff                        = (i) + (k)*nx_g + j * nx_g * nz_g;
          buffer[buffer_start + indx_buff] = Grav.F.potential_h[indx];
        }
      }
    }
  }

  // Load Z boundaries
  if (direction == 2) {
    length = nGHST * nx_g * ny_g;
    for (k = 0; k < nGHST; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx = (i) + (j)*nx_g + (k + nGHST) * nx_g * ny_g;
          }
          if (side == 1) {
            indx = (i) + (j)*nx_g + (nz_g - 2 * nGHST + k) * nx_g * ny_g;
          }
          indx_buff                        = (i) + (j)*nx_g + k * nx_g * ny_g;
          buffer[buffer_start + indx_buff] = Grav.F.potential_h[indx];
        }
      }
    }
  }
  return length;
}

void Grid3D::Unload_Gravity_Potential_from_Buffer(int direction, int side, Real *buffer, int buffer_start)
{
  int i, j, k, indx, indx_buff;
  int nGHST, nx_g, ny_g, nz_g;
  nGHST = N_GHOST_POTENTIAL;
  nx_g  = Grav.nx_local + 2 * nGHST;
  ny_g  = Grav.ny_local + 2 * nGHST;
  nz_g  = Grav.nz_local + 2 * nGHST;

  // Load X boundaries
  if (direction == 0) {
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nGHST; i++) {
          if (side == 0) {
            indx = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx = (nx_g - nGHST + i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          indx_buff                = (j) + (k)*ny_g + i * ny_g * nz_g;
          Grav.F.potential_h[indx] = buffer[buffer_start + indx_buff];
        }
      }
    }
  }

  // Load Y boundaries
  if (direction == 1) {
    for (k = 0; k < nz_g; k++) {
      for (j = 0; j < nGHST; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx = (i) + (ny_g - nGHST + j) * nx_g + (k)*nx_g * ny_g;
          }
          indx_buff                = (i) + (k)*nx_g + j * nx_g * nz_g;
          Grav.F.potential_h[indx] = buffer[buffer_start + indx_buff];
        }
      }
    }
  }

  // Load Z boundaries
  if (direction == 2) {
    for (k = 0; k < nGHST; k++) {
      for (j = 0; j < ny_g; j++) {
        for (i = 0; i < nx_g; i++) {
          if (side == 0) {
            indx = (i) + (j)*nx_g + (k)*nx_g * ny_g;
          }
          if (side == 1) {
            indx = (i) + (j)*nx_g + (nz_g - nGHST + k) * nx_g * ny_g;
          }
          indx_buff                = (i) + (j)*nx_g + k * nx_g * ny_g;
          Grav.F.potential_h[indx] = buffer[buffer_start + indx_buff];
        }
      }
    }
  }
}

  #endif  // GRAVITY
#endif    // MPI_CHOLLA
