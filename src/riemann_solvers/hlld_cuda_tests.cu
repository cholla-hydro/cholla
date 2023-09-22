/*!
 * \file hlld_cuda_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Test the code units within hlld_cuda.cu
 *
 */

// STL Includes
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global_cuda.h"
#include "../grid/grid_enum.h"
#include "../riemann_solvers/hlld_cuda.h"  // Include code to test
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"
#include "../utils/mhd_utilities.h"
#include "../utils/testing_utilities.h"

#ifdef CUDA
  #ifdef MHD
// =========================================================================
// Integration tests for the entire HLLD solver. Unit tests are below
// =========================================================================

// =========================================================================
/*!
* \brief Test fixture for simple testing of the HLLD Riemann Solver.
Effectively takes the left state, right state, fiducial fluxes, and
custom user output then performs all the required running and testing
*
*/
// NOLINTNEXTLINE(readability-identifier-naming)
class tMHDCalculateHLLDFluxesCUDA : public ::testing::Test
{
 protected:
  // =====================================================================
  /*!
   * \brief Compute and return the HLLD fluxes
   *
   * \param[in] leftState The state on the left side in conserved
   * variables. In order the elements are: density, x-momentum,
   * y-momentum, z-momentum, energy, passive scalars, x-magnetic field,
   * y-magnetic field, z-magnetic field.
   * \param[in] rightState The state on the right side in conserved
   * variables. In order the elements are: density, x-momentum,
   * y-momentum, z-momentum, energy, passive scalars, x-magnetic field,
   * y-magnetic field, z-magnetic field.
   * \param[in] gamma The adiabatic index
   * \param[in] direction Which plane the interface is. 0 = plane normal to
   * X, 1 = plane normal to Y, 2 = plane normal to Z. Defaults to 0.
   * \return std::vector<double>
   */
  std::vector<Real> Compute_Fluxes(std::vector<Real> stateLeft, std::vector<Real> stateRight, Real const &gamma,
                                   int const &direction = 0)
  {
    // Rearrange X, Y, and Z values for the chosen direction
    std::rotate(stateLeft.begin() + 1, stateLeft.begin() + 4 - direction, stateLeft.begin() + 4);
    std::rotate(stateRight.begin() + 1, stateRight.begin() + 4 - direction, stateRight.begin() + 4);

    // Create new vectors that store the values in the way that the HLLD
    // solver expects
    EXPECT_DOUBLE_EQ(stateLeft.at(grid_enum::magnetic_x), stateRight.at(grid_enum::magnetic_x))
        << "The left and right magnetic fields are not equal";
    std::vector<Real> const magneticX{stateLeft.at(grid_enum::magnetic_x)};
    stateLeft.erase(stateLeft.begin() + grid_enum::magnetic_x);
    stateRight.erase(stateRight.begin() + grid_enum::magnetic_x);

    // Simulation Paramters
    int const nx      = 1;  // Number of cells in the x-direction
    int const ny      = 1;  // Number of cells in the y-direction
    int const nz      = 1;  // Number of cells in the z-direction
    int const n_cells = nx * ny * nz;
    int nFields       = 8;  // Total number of conserved fields
    #ifdef SCALAR
    nFields += NSCALARS;
    #endif  // SCALAR
    #ifdef DE
    nFields++;
    #endif  // DE

    // Launch Parameters
    dim3 const dimGrid(1, 1, 1);   // How many blocks in the grid
    dim3 const dimBlock(1, 1, 1);  // How many threads per block

    // Create the std::vector to store the fluxes and declare the device
    // pointers
    std::vector<Real> testFlux(nFields - 1, 0);
    Real *devConservedLeft;
    Real *devConservedRight;
    Real *devConservedMagXFace;
    Real *devTestFlux;

    // Allocate device arrays and copy data
    CudaSafeCall(cudaMalloc(&devConservedLeft, stateLeft.size() * sizeof(Real)));
    CudaSafeCall(cudaMalloc(&devConservedRight, stateRight.size() * sizeof(Real)));
    CudaSafeCall(cudaMalloc(&devConservedMagXFace, magneticX.size() * sizeof(Real)));
    CudaSafeCall(cudaMalloc(&devTestFlux, testFlux.size() * sizeof(Real)));

    CudaSafeCall(
        cudaMemcpy(devConservedLeft, stateLeft.data(), stateLeft.size() * sizeof(Real), cudaMemcpyHostToDevice));
    CudaSafeCall(
        cudaMemcpy(devConservedRight, stateRight.data(), stateRight.size() * sizeof(Real), cudaMemcpyHostToDevice));
    CudaSafeCall(
        cudaMemcpy(devConservedMagXFace, magneticX.data(), magneticX.size() * sizeof(Real), cudaMemcpyHostToDevice));

    // Run kernel
    hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, dimGrid, dimBlock, 0, 0,
                       devConservedLeft,      // the "left" interface
                       devConservedRight,     // the "right" interface
                       devConservedMagXFace,  // the magnetic field at the interface
                       devTestFlux, n_cells, gamma, direction, nFields);

    CudaCheckError();
    CudaSafeCall(cudaMemcpy(testFlux.data(), devTestFlux, testFlux.size() * sizeof(Real), cudaMemcpyDeviceToHost));

    // Make sure to sync with the device so we have the results
    cudaDeviceSynchronize();
    CudaCheckError();

    // Free device arrays
    cudaFree(devConservedLeft);
    cudaFree(devConservedRight);
    cudaFree(devConservedMagXFace);
    cudaFree(devTestFlux);

    // The HLLD solver only writes the the first two "slots" for
    // magnetic flux so let's rearrange to make sure we have all the
    // magnetic fluxes in the right spots
    testFlux.insert(testFlux.begin() + grid_enum::magnetic_x, 0.0);
    std::rotate(testFlux.begin() + 1, testFlux.begin() + 1 + direction,
                testFlux.begin() + 4);  // Rotate momentum

    return testFlux;
  }
  // =====================================================================

  // =====================================================================
  /*!
   * \brief Check if the fluxes are correct
   *
   * \param[in] fiducialFlux The fiducial flux in conserved variables. In
   * order the elements are: density, x-momentum,
   * y-momentum, z-momentum, energy, passive scalars, x-magnetic field,
   * y-magnetic field, z-magnetic field.
   * \param[in] scalarFlux The fiducial flux in the passive scalars
   * \param[in] thermalEnergyFlux The fiducial flux in the dual energy
   * thermal energy
   * \param[in] testFlux The test flux in conserved variables. In order the
   * elements are: density, x-momentum,
   * y-momentum, z-momentum, energy, passive scalars, x-magnetic field,
   * y-magnetic field, z-magnetic field.
   * \param[in] customOutput Any custom output the user would like to
   * print. It will print after the default GTest output but before the
   * values that failed are printed
   * \param[in] direction Which plane the interface is. 0 = plane normal to
   * X, 1 = plane normal to Y, 2 = plane normal to Z. Defaults to 0.
   */
  void Check_Results(std::vector<Real> fiducialFlux, std::vector<Real> const &scalarFlux, Real thermalEnergyFlux,
                     std::vector<Real> const &testFlux, std::string const &customOutput = "", int const &direction = 0)
  {
    // Field names
    std::vector<std::string> fieldNames{"Densities", "X Momentum",       "Y Momentum",       "Z Momentum",
                                        "Energies",  "X Magnetic Field", "Y Magnetic Field", "Z Magnetic Field"};
    #ifdef DE
    fieldNames.push_back("Thermal energy (dual energy)");
    fiducialFlux.push_back(thermalEnergyFlux);
    #endif  // DE
    #ifdef SCALAR
    std::vector<std::string> scalarNames{"Scalar 1", "Scalar 2", "Scalar 3"};
    fieldNames.insert(fieldNames.begin() + grid_enum::magnetic_start, scalarNames.begin(),
                      scalarNames.begin() + grid_enum::nscalars);

    fiducialFlux.insert(fiducialFlux.begin() + grid_enum::magnetic_start, scalarFlux.begin(),
                        scalarFlux.begin() + grid_enum::nscalars);
    #endif  // SCALAR

    ASSERT_TRUE((fiducialFlux.size() == testFlux.size()) and (fiducialFlux.size() == fieldNames.size()))
        << "The fiducial flux, test flux, and field name vectors are not all "
           "the same length"
        << std::endl
        << "fiducialFlux.size() = " << fiducialFlux.size() << std::endl
        << "testFlux.size() = " << testFlux.size() << std::endl
        << "fieldNames.size() = " << fieldNames.size() << std::endl;

    // Check for equality
    for (size_t i = 0; i < fieldNames.size(); i++) {
      // Check for equality and if not equal return difference
      double absoluteDiff;
      int64_t ulpsDiff;

      // This error is consistent with the FP error in rearanging the flux
      // computations in the Athena solver
      double const fixedEpsilon = 2.7E-15;
      int64_t const ulpsEpsilon = 7;

      bool areEqual = testing_utilities::nearlyEqualDbl(fiducialFlux[i], testFlux[i], absoluteDiff, ulpsDiff,
                                                        fixedEpsilon, ulpsEpsilon);
      EXPECT_TRUE(areEqual) << std::endl
                            << customOutput << std::endl
                            << "There's a difference in " << fieldNames[i] << " Flux" << std::endl
                            << "The direction is:       " << direction << " (0=X, 1=Y, 2=Z)" << std::endl
                            << "The fiducial value is:       " << fiducialFlux[i] << std::endl
                            << "The test value is:           " << testFlux[i] << std::endl
                            << "The absolute difference is:  " << absoluteDiff << std::endl
                            << "The ULP difference is:       " << ulpsDiff << std::endl;
    }
  }
  // =====================================================================

  // =====================================================================
  /*!
   * \brief Convert a vector of quantities in primitive variables  to
   * conserved variables
   *
   * \param[in] input The state in primitive variables. In order the
   * elements are: density, x-momentum,
   * y-momentum, z-momentum, energy, passive scalars, x-magnetic field,
   * y-magnetic field, z-magnetic field.
   * \return std::vector<Real> The state in conserved variables. In order
   * the elements are: density, x-momentum,
   * y-momentum, z-momentum, energy, passive scalars, x-magnetic field,
   * y-magnetic field, z-magnetic field.
   */
  std::vector<Real> Primitive_2_Conserved(std::vector<Real> const &input, double const &gamma,
                                          std::vector<Real> const &primitiveScalars)
  {
    std::vector<Real> output(input.size());
    output.at(0) = input.at(0);                // Density
    output.at(1) = input.at(1) * input.at(0);  // X Velocity to momentum
    output.at(2) = input.at(2) * input.at(0);  // Y Velocity to momentum
    output.at(3) = input.at(3) * input.at(0);  // Z Velocity to momentum
    output.at(4) =
        hydro_utilities::Calc_Energy_Primitive(input.at(4), input.at(0), input.at(1), input.at(2), input.at(3), gamma,
                                               input.at(5), input.at(6), input.at(7));  // Pressure to Energy
    output.at(5) = input.at(5);                                                         // X Magnetic Field
    output.at(6) = input.at(6);                                                         // Y Magnetic Field
    output.at(7) = input.at(7);                                                         // Z Magnetic Field

    #ifdef SCALAR
    std::vector<Real> conservedScalar(primitiveScalars.size());
    std::transform(primitiveScalars.begin(), primitiveScalars.end(), conservedScalar.begin(),
                   [&](Real const &c) { return c * output.at(0); });
    output.insert(output.begin() + grid_enum::magnetic_start, conservedScalar.begin(),
                  conservedScalar.begin() + grid_enum::nscalars);
    #endif  // SCALAR
    #ifdef DE
    output.push_back(mhd::utils::computeThermalEnergy(
        output.at(4), output.at(0), output.at(1), output.at(2), output.at(3), output.at(grid_enum::magnetic_x),
        output.at(grid_enum::magnetic_y), output.at(grid_enum::magnetic_z), gamma));
    #endif  // DE
    return output;
  }
  // =====================================================================

  // =====================================================================
  /*!
   * \brief On test start make sure that the number of NSCALARS is allowed
   *
   */
  void SetUp()
  {
    #ifdef SCALAR
    ASSERT_LE(NSCALARS, 3) << "Only up to 3 passive scalars are currently "
                              "supported in HLLD tests. NSCALARS = "
                           << NSCALARS;
    ASSERT_GE(NSCALARS, 1) << "There must be at least 1 passive scalar to test "
                              "with passive scalars. NSCALARS = "
                           << NSCALARS;
    #endif  // SCALAR
  }
  // =====================================================================
 private:
};
// =========================================================================

// =========================================================================
/*!
 * \brief Test the HLLD Riemann Solver using various states and waves from
 * the Brio & Wu Shock tube
 *
 */
TEST_F(tMHDCalculateHLLDFluxesCUDA, BrioAndWuShockTubeCorrectInputExpectCorrectOutput)
{
  // Constant Values
  Real const gamma = 2.;
  Real const Vz    = 0.0;
  Real const Bx    = 0.75;
  Real const Bz    = 0.0;
  std::vector<Real> const primitiveScalar{1.1069975296, 2.2286185018, 3.3155141875};

  // States
  std::vector<Real> const  // | Density | X-Velocity | Y-Velocity | Z-Velocity |
                           // Pressure | X-Magnetic Field | Y-Magnetic Field |
                           // Z-Magnetic Field | Adiabatic Index | Passive
                           // Scalars |
      leftICs               = Primitive_2_Conserved({1.0, 0.0, 0.0, Vz, 1.0, Bx, 1.0, Bz}, gamma, primitiveScalar),
      leftFastRareLeftSide  = Primitive_2_Conserved({0.978576, 0.038603, -0.011074, Vz, 0.957621, Bx, 0.970288, Bz},
                                                    gamma, primitiveScalar),
      leftFastRareRightSide = Primitive_2_Conserved({0.671655, 0.647082, -0.238291, Vz, 0.451115, Bx, 0.578240, Bz},
                                                    gamma, primitiveScalar),
      compoundLeftSide  = Primitive_2_Conserved({0.814306, 0.506792, -0.911794, Vz, 0.706578, Bx, -0.108819, Bz}, gamma,
                                                primitiveScalar),
      compoundPeak      = Primitive_2_Conserved({0.765841, 0.523701, -1.383720, Vz, 0.624742, Bx, -0.400787, Bz}, gamma,
                                                primitiveScalar),
      compoundRightSide = Primitive_2_Conserved({0.695211, 0.601089, -1.583720, Vz, 0.515237, Bx, -0.537027, Bz}, gamma,
                                                primitiveScalar),
      contactLeftSide   = Primitive_2_Conserved({0.680453, 0.598922, -1.584490, Vz, 0.515856, Bx, -0.533616, Bz}, gamma,
                                                primitiveScalar),
      contactRightSide  = Primitive_2_Conserved({0.231160, 0.599261, -1.584820, Vz, 0.516212, Bx, -0.533327, Bz}, gamma,
                                                primitiveScalar),
      slowShockLeftSide = Primitive_2_Conserved({0.153125, 0.086170, -0.683303, Vz, 0.191168, Bx, -0.850815, Bz}, gamma,
                                                primitiveScalar),
      slowShockRightSide     = Primitive_2_Conserved({0.117046, -0.238196, -0.165561, Vz, 0.087684, Bx, -0.903407, Bz},
                                                     gamma, primitiveScalar),
      rightFastRareLeftSide  = Primitive_2_Conserved({0.117358, -0.228756, -0.158845, Vz, 0.088148, Bx, -0.908335, Bz},
                                                     gamma, primitiveScalar),
      rightFastRareRightSide = Primitive_2_Conserved({0.124894, -0.003132, -0.002074, Vz, 0.099830, Bx, -0.999018, Bz},
                                                     gamma, primitiveScalar),
      rightICs               = Primitive_2_Conserved({0.128, 0.0, 0.0, Vz, 0.1, Bx, -1.0, Bz}, gamma, primitiveScalar);

  for (size_t direction = 0; direction < 3; direction++) {
    // Initial Condition Checks
    {
      std::string const outputString{
          "Left State:  Left Brio & Wu state\n"
          "Right State: Left Brio & Wu state\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0, 1.21875, -0.75, 0, 0, 0.0, 0, 0};
      std::vector<Real> const scalarFlux{0, 0, 0};
      Real thermalEnergyFlux             = 0.0;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftICs, leftICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Brio & Wu state\n"
          "Right State: Right Brio & Wu state\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0, 0.31874999999999998, 0.75, 0, 0, 0.0, 0, 0};
      std::vector<Real> const scalarFlux{0, 0, 0};
      Real thermalEnergyFlux             = 0.0;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightICs, rightICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left Brio & Wu state\n"
          "Right State: Right Brio & Wu state\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.20673357746080057,  0.4661897584603672,
                                           0.061170028480309613, 0,
                                           0.064707291981509041, 0.0,
                                           1.0074980455427278,   0};
      std::vector<Real> const scalarFlux{0.22885355953447648, 0.46073027567244362, 0.6854281091039145};
      Real thermalEnergyFlux             = 0.20673357746080046;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftICs, rightICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left Brio & Wu state with negative Bx\n"
          "Right State: Right Brio & Wu state with negative Bx\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.20673357746080057,   0.4661897584603672,
                                           -0.061170028480309613, 0,
                                           0.064707291981509041,  0.0,
                                           1.0074980455427278,    0};
      std::vector<Real> const scalarFlux{0.22885355953447648, 0.46073027567244362, 0.6854281091039145};
      Real thermalEnergyFlux = 0.20673357746080046;

      std::vector<Real> leftICsNegBx = leftICs, rightICsNegBx = rightICs;
      leftICsNegBx[5]  = -leftICsNegBx[5];
      rightICsNegBx[5] = -rightICsNegBx[5];

      std::vector<Real> const testFluxes = Compute_Fluxes(leftICsNegBx, rightICsNegBx, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Brio & Wu state\n"
          "Right State: Left Brio & Wu state\n"
          "HLLD State: Right Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-0.20673357746080057,  0.4661897584603672,
                                           0.061170028480309613,  0,
                                           -0.064707291981509041, 0.0,
                                           -1.0074980455427278,   0};
      std::vector<Real> const scalarFlux{-0.22885355953447648, -0.46073027567244362, -0.6854281091039145};
      Real thermalEnergyFlux             = -0.20673357746080046;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightICs, leftICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }

    // Cross wave checks
    {
      std::string const outputString{
          "Left State:  Left of left fast rarefaction\n"
          "Right State: Right of left fast rarefaction\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.4253304970883941,   0.47729308161522394,
                                           -0.55321646324583107, 0,
                                           0.92496835095531071,  0.0,
                                           0.53128887284876058,  0};
      std::vector<Real> const scalarFlux{0.47083980954039228, 0.94789941519098619, 1.4101892974729979};
      Real thermalEnergyFlux = 0.41622256825457099;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(leftFastRareLeftSide, leftFastRareRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of left fast rarefaction\n"
          "Right State: Left of left fast rarefaction\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.070492123816403796, 1.2489600267034342,
                                           -0.71031457071286608, 0,
                                           0.21008080091470105,  0.0,
                                           0.058615131833681167, 0};
      std::vector<Real> const scalarFlux{0.078034606921016325, 0.15710005136841393, 0.23371763662029341};
      Real thermalEnergyFlux = 0.047345816580591255;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(leftFastRareRightSide, leftFastRareLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of compound wave\n"
          "Right State: Right of compound wave\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.4470171023231666,   0.60747660800918468,
                                           -0.20506357956052623, 0,
                                           0.72655525704800772,  0.0,
                                           0.76278089951123285,  0};
      std::vector<Real> const scalarFlux{0.4948468279606959, 0.99623058485843297, 1.482091544807598};
      Real thermalEnergyFlux             = 0.38787931087981475;
      std::vector<Real> const testFluxes = Compute_Fluxes(compoundLeftSide, compoundRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of compound wave\n"
          "Right State: Left of compound wave\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.38496850292724116, 0.66092864409611585,
                                           -0.3473204105316457, 0,
                                           0.89888639514227009, 0.0,
                                           0.71658566275120927, 0};
      std::vector<Real> const scalarFlux{0.42615918171426637, 0.85794792823389721, 1.2763685331959034};
      Real thermalEnergyFlux             = 0.28530908823756074;
      std::vector<Real> const testFluxes = Compute_Fluxes(compoundRightSide, compoundLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of Compound Wave\n"
          "Right State: Peak of Compound Wave\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.41864266180405574, 0.63505764056357727,
                                           -0.1991008813536404, 0,
                                           0.73707474818824525, 0.0,
                                           0.74058225030218761, 0};
      std::vector<Real> const scalarFlux{0.46343639240225803, 0.93299478173931882, 1.388015684704111};
      Real thermalEnergyFlux             = 0.36325864563467081;
      std::vector<Real> const testFluxes = Compute_Fluxes(compoundLeftSide, compoundPeak, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Peak of Compound Wave\n"
          "Right State: Left of Compound Wave\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.39520761138156862,  0.6390998385557225,
                                           -0.35132701297727598, 0,
                                           0.89945171879176522,  0.0,
                                           0.71026545717401468,  0};
      std::vector<Real> const scalarFlux{0.43749384947851333, 0.88076699477714815, 1.3103164425435772};
      Real thermalEnergyFlux             = 0.32239432669410983;
      std::vector<Real> const testFluxes = Compute_Fluxes(compoundPeak, compoundLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Peak of Compound Wave\n"
          "Right State: Right of Compound Wave\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.4285899590904928,   0.6079309920345296,
                                           -0.26055320217638239, 0,
                                           0.75090757444649436,  0.0,
                                           0.85591904930227747,  0};
      std::vector<Real> const scalarFlux{0.47444802592454061, 0.95516351251477749, 1.4209960899845735};
      Real thermalEnergyFlux             = 0.34962629086469987;
      std::vector<Real> const testFluxes = Compute_Fluxes(compoundPeak, compoundRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of Compound Wave\n"
          "Right State: Peak of Compound Wave\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.39102247793946454,  0.65467021266207581,
                                           -0.25227691377588229, 0,
                                           0.76271525822813691,  0.0,
                                           0.83594460438033491,  0};
      std::vector<Real> const scalarFlux{0.43286091709705776, 0.8714399289555731, 1.2964405732397004};
      Real thermalEnergyFlux             = 0.28979582956267347;
      std::vector<Real> const testFluxes = Compute_Fluxes(compoundRightSide, compoundPeak, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of contact discontinuity\n"
          "Right State: Right of contact discontinuity\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.40753761783585118, 0.62106392255463172,
                                           -0.2455554035355339, 0,
                                           0.73906344777217226, 0.0,
                                           0.8687394222350926,  0};
      std::vector<Real> const scalarFlux{0.45114313616335622, 0.90824587528847567, 1.3511967538747176};
      Real thermalEnergyFlux             = 0.30895701155896288;
      std::vector<Real> const testFluxes = Compute_Fluxes(contactLeftSide, contactRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of contact discontinuity\n"
          "Right State: Left of contact discontinuity\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.13849588572126192, 0.46025037934770729,
                                           0.18052412687974539, 0,
                                           0.35385590617992224, 0.0,
                                           0.86909622543144227, 0};
      std::vector<Real> const scalarFlux{0.15331460335320088, 0.30865449334158279, 0.45918507401922254};
      Real thermalEnergyFlux             = 0.30928031735570188;
      std::vector<Real> const testFluxes = Compute_Fluxes(contactRightSide, contactLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Slow shock left side\n"
          "Right State: Slow shock right side\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{3.5274134848883865e-05, 0.32304849716274459,
                                           0.60579784881286636,    0,
                                           -0.32813070621836449,   0.0,
                                           0.40636483121437972,    0};
      std::vector<Real> const scalarFlux{3.9048380136491711e-05, 7.8612589559210735e-05, 0.00011695189454326261};
      Real thermalEnergyFlux             = 4.4037784886918126e-05;
      std::vector<Real> const testFluxes = Compute_Fluxes(slowShockLeftSide, slowShockRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Slow shock right side\n"
          "Right State: Slow shock left side\n"
          "HLLD State: Right Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-0.016514307834939734, 0.16452009375678914,
                                           0.71622171077118635,   0,
                                           -0.37262428139914472,  0.0,
                                           0.37204015363322052,   0};
      std::vector<Real> const scalarFlux{-0.018281297976332211, -0.036804091985367396, -0.054753421923485097};
      Real thermalEnergyFlux             = -0.020617189878790236;
      std::vector<Real> const testFluxes = Compute_Fluxes(slowShockRightSide, slowShockLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right fast rarefaction left side\n"
          "Right State: Right fast rarefaction right side\n"
          "HLLD State: Right Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-0.026222824218991747, 0.22254903570732654,
                                           0.68544334213642255,   0,
                                           -0.33339172106895454,  0.0,
                                           0.32319665359522443,   0};
      std::vector<Real> const scalarFlux{-0.029028601629558917, -0.058440671223894146, -0.086942145734385745};
      Real thermalEnergyFlux = -0.020960370728633469;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(rightFastRareLeftSide, rightFastRareRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right fast rarefaction right side\n"
          "Right State: Right fast rarefaction left side\n"
          "HLLD State: Right Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-0.001088867226159973,  0.32035322820305906,
                                           0.74922357263343131,    0,
                                           -0.0099746892805345766, 0.0,
                                           0.0082135595470345102,  0};
      std::vector<Real> const scalarFlux{-0.0012053733294214947, -0.0024266696462237609, -0.0036101547366371614};
      Real thermalEnergyFlux = -0.00081785194236053073;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(rightFastRareRightSide, rightFastRareLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the HLLD Riemann Solver using various states and waves from
 * the Dai & Woodward Shock tube
 *
 */
TEST_F(tMHDCalculateHLLDFluxesCUDA, DaiAndWoodwardShockTubeCorrectInputExpectCorrectOutput)
{
  // Constant Values
  Real const gamma = 5. / 3.;
  Real const coef  = 1. / (std::sqrt(4. * M_PI));
  Real const Bx    = 4. * coef;
  std::vector<Real> const primitiveScalar{1.1069975296, 2.2286185018, 3.3155141875};

  // States
  std::vector<Real> const  // | Density | X-Velocity | Y-Velocity | Z-Velocity |
                           // Pressure | X-Magnetic Field | Y-Magnetic Field |
                           // Z-Magnetic Field | Adiabatic Index | Passive Scalars |
      leftICs = Primitive_2_Conserved({1.08, 0.0, 0.0, 0.0, 1.0, Bx, 3.6 * coef, 2 * coef}, gamma, primitiveScalar),
      leftFastShockLeftSide = Primitive_2_Conserved(
          {1.09406, 1.176560, 0.021003, 0.506113, 0.970815, 1.12838, 1.105355, 0.614087}, gamma, primitiveScalar),
      leftFastShockRightSide = Primitive_2_Conserved(
          {1.40577, 0.693255, 0.210562, 0.611423, 1.494290, 1.12838, 1.457700, 0.809831}, gamma, primitiveScalar),
      leftRotationLeftSide = Primitive_2_Conserved(
          {1.40086, 0.687774, 0.215124, 0.609161, 1.485660, 1.12838, 1.458735, 0.789960}, gamma, primitiveScalar),
      leftRotationRightSide = Primitive_2_Conserved(
          {1.40119, 0.687504, 0.330268, 0.334140, 1.486570, 1.12838, 1.588975, 0.475782}, gamma, primitiveScalar),
      leftSlowShockLeftSide = Primitive_2_Conserved(
          {1.40519, 0.685492, 0.326265, 0.333664, 1.493710, 1.12838, 1.575785, 0.472390}, gamma, primitiveScalar),
      leftSlowShockRightSide = Primitive_2_Conserved(
          {1.66488, 0.578545, 0.050746, 0.250260, 1.984720, 1.12838, 1.344490, 0.402407}, gamma, primitiveScalar),
      contactLeftSide = Primitive_2_Conserved(
          {1.65220, 0.578296, 0.049683, 0.249962, 1.981250, 1.12838, 1.346155, 0.402868}, gamma, primitiveScalar),
      contactRightSide = Primitive_2_Conserved(
          {1.49279, 0.578276, 0.049650, 0.249924, 1.981160, 1.12838, 1.346180, 0.402897}, gamma, primitiveScalar),
      rightSlowShockLeftSide = Primitive_2_Conserved(
          {1.48581, 0.573195, 0.035338, 0.245592, 1.956320, 1.12838, 1.370395, 0.410220}, gamma, primitiveScalar),
      rightSlowShockRightSide = Primitive_2_Conserved(
          {1.23813, 0.450361, -0.275532, 0.151746, 1.439000, 1.12838, 1.609775, 0.482762}, gamma, primitiveScalar),
      rightRotationLeftSide = Primitive_2_Conserved(
          {1.23762, 0.450102, -0.274410, 0.145585, 1.437950, 1.12838, 1.606945, 0.493879}, gamma, primitiveScalar),
      rightRotationRightSide = Primitive_2_Conserved(
          {1.23747, 0.449993, -0.180766, -0.090238, 1.437350, 1.12838, 1.503855, 0.752090}, gamma, primitiveScalar),
      rightFastShockLeftSide = Primitive_2_Conserved(
          {1.22305, 0.424403, -0.171402, -0.085701, 1.409660, 1.12838, 1.447730, 0.723864}, gamma, primitiveScalar),
      rightFastShockRightSide = Primitive_2_Conserved(
          {1.00006, 0.000121, -0.000057, -0.000028, 1.000100, 1.12838, 1.128435, 0.564217}, gamma, primitiveScalar),
      rightICs = Primitive_2_Conserved({1.0, 0.0, 0.0, 1.0, 0.2, Bx, 4 * coef, 2 * coef}, gamma, primitiveScalar);

  for (size_t direction = 0; direction < 3; direction++) {
    // Initial Condition Checks
    {
      std::string const outputString{
          "Left State:  Left Dai & Woodward state\n"
          "Right State: Left Dai & Woodward state\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0, 1.0381971863420549,     -1.1459155902616465, -0.63661977236758127, 0, 0.0,
                                           0, -1.1102230246251565e-16};
      std::vector<Real> const scalarFlux{0, 0, 0};
      Real thermalEnergyFlux             = 0.0;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftICs, leftICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Dai & Woodward state\n"
          "Right State: Right Dai & Woodward state\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          0,   0.35915494309189522,    -1.2732395447351625, -0.63661977236758127, -0.63661977236758172,
          0.0, 2.2204460492503131e-16, -1.1283791670955123};
      std::vector<Real> const scalarFlux{0, 0, 0};
      Real thermalEnergyFlux             = 0.0;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightICs, rightICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left Dai & Woodward state\n"
          "Right State: Right Dai & Woodward state\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.17354924587196074,  0.71614983677687327,  -1.1940929411768009,
                                           -1.1194725181819352,  -0.11432087006939984, 0.0,
                                           0.056156000248263505, -0.42800560867873094};
      std::vector<Real> const scalarFlux{0.19211858644420357, 0.38677506032368902, 0.57540498691841158};
      Real thermalEnergyFlux             = 0.24104061926661174;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftICs, rightICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Dai & Woodward state\n"
          "Right State: Left Dai & Woodward state\n"
          "HLLD State: Right Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-0.17354924587196074,  0.71614983677687327,  -1.1940929411768009,
                                           -0.14549552299758384,  -0.47242308031148195, 0.0,
                                           -0.056156000248263505, -0.55262526758377528};
      std::vector<Real> const scalarFlux{-0.19211858644420357, -0.38677506032368902, -0.57540498691841158};
      Real thermalEnergyFlux             = -0.24104061926661174;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightICs, leftICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }

    // Cross wave checks
    {
      std::string const outputString{
          "Left State:  Left of left fast shock\n"
          "Right State: Right of left fast shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.96813688187727132,  3.0871217875403394,   -1.4687093290523414,
                                           -0.33726008721080036, 4.2986213406773457,   0.0,
                                           0.84684181393860269,  -0.087452560407274671};
      std::vector<Real> const scalarFlux{1.0717251365527865, 2.157607767226648, 3.2098715673061045};
      Real thermalEnergyFlux = 1.2886155333980993;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(leftFastShockLeftSide, leftFastShockRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of left fast shock\n"
          "Right State: Left of left fast shock\n"
          "HLLD State: Left Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{1.3053938862274184,    2.4685129176021858, -1.181892850065283,
                                           -0.011160487372167127, 5.1797404608257249, 0.0,
                                           1.1889903073770265,    0.10262704114294516};
      std::vector<Real> const scalarFlux{1.4450678072086958, 2.9092249669830292, 4.3280519500627666};
      Real thermalEnergyFlux = 2.081389946702628;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(leftFastShockRightSide, leftFastShockLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of left rotation/Alfven wave\n"
          "Right State: Right of left rotation/Alfven wave\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.96326128304298586,  2.8879592118317445,   -1.4808188010794987,
                                           -0.20403672861184916, 4.014027751838869,    0.0,
                                           0.7248753989305099,   -0.059178137562467162};
      std::vector<Real> const scalarFlux{1.0663278606879119, 2.1467419174572049, 3.1937064501984724};
      Real thermalEnergyFlux = 1.5323573637968553;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(leftRotationLeftSide, leftRotationRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of left rotation/Alfven wave\n"
          "Right State: Left of left rotation/Alfven wave\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.96353754504060063,  2.8875487093397085,  -1.4327309336053695,
                                           -0.31541343522923493, 3.9739842521208342,  0.0,
                                           0.75541746728406312,  -0.13479771672887678};
      std::vector<Real> const scalarFlux{1.0666336820367937, 2.1473576000564334, 3.1946224007710313};
      Real thermalEnergyFlux = 1.5333744977458499;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(leftRotationRightSide, leftRotationLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of left slow shock\n"
          "Right State: Right of left slow shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.88716095730727451,  2.9828594399125663,  -1.417062582518549,
                                           -0.21524331343191233, 3.863474778369334,   0.0,
                                           0.71242370728996041,  -0.05229712416644372};
      std::vector<Real> const scalarFlux{0.98208498809672407, 1.9771433235295921, 2.9413947405483505};
      Real thermalEnergyFlux = 1.4145715457049737;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(leftSlowShockLeftSide, leftSlowShockRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of left slow shock\n"
          "Right State: Left of left slow shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{1.042385440439527,    2.7732383399777376,   -1.5199872074603551,
                                           -0.21019362664841068, 4.1322001036232585,   0.0,
                                           0.72170937317481543,  -0.049474715634396704};
      std::vector<Real> const scalarFlux{1.1539181074575644, 2.323079478570472, 3.4560437166206879};
      Real thermalEnergyFlux = 1.8639570701934713;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(leftSlowShockRightSide, leftSlowShockLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of contact discontinuity\n"
          "Right State: Right of contact discontinuity\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.95545795601418737,  2.8843900822429749,   -1.4715039715239722,
                                           -0.21575736014726318, 4.0078718055059257,   0.0,
                                           0.72241353110189066,  -0.049073560388753337};
      std::vector<Real> const scalarFlux{1.0576895969443709, 2.1293512784652289, 3.1678344087247892};
      Real thermalEnergyFlux             = 1.7186185770667382;
      std::vector<Real> const testFluxes = Compute_Fluxes(contactLeftSide, contactRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of contact discontinuity\n"
          "Right State: Left of contact discontinuity\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.86324813554422819,  2.8309913324581251,   -1.4761428591480787,
                                           -0.23887765947428419, 3.9892942559102793,   0.0,
                                           0.72244123046603836,  -0.049025527032060034};
      std::vector<Real> const scalarFlux{0.95561355347926669, 1.9238507665182214, 2.8621114407298114};
      Real thermalEnergyFlux             = 1.7184928987481187;
      std::vector<Real> const testFluxes = Compute_Fluxes(contactRightSide, contactLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of right slow shock\n"
          "Right State: Right of right slow shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.81125524370350677, 2.901639500435365,    -1.5141545346789429,
                                           -0.262600896007809,  3.8479660419540087,   0.0,
                                           0.7218977970017596,  -0.049091614519593846};
      std::vector<Real> const scalarFlux{0.89805755065482806, 1.8079784457999033, 2.6897282701827465};
      Real thermalEnergyFlux = 1.6022319728249694;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(rightSlowShockLeftSide, rightSlowShockRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of right slow shock\n"
          "Right State: Left of right slow shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.60157947557836688,  2.3888357198399746, -1.9910500022202977,
                                           -0.45610948442354332, 3.5359430988850069, 0.0,
                                           1.0670963294022622,   0.05554893654378229};
      std::vector<Real> const scalarFlux{0.66594699332331575, 1.3406911495770899, 1.994545286188885};
      Real thermalEnergyFlux = 1.0487665253534804;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(rightSlowShockRightSide, rightSlowShockLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of right rotation/Alfven wave\n"
          "Right State: Right of right rotation/Alfven wave\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.55701691287884714,  2.4652223621237814,  -1.9664615862227277,
                                           -0.47490477894092042, 3.3900659850690529,  0.0,
                                           1.0325648885587542,   0.059165409025635551};
      std::vector<Real> const scalarFlux{0.61661634650230224, 1.2413781978573175, 1.8467974773272691};
      Real thermalEnergyFlux = 0.9707694646266285;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(rightRotationLeftSide, rightRotationRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of right rotation/Alfven wave\n"
          "Right State: Left of right rotation/Alfven wave\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.55689116371132596,  2.4648517303940851, -1.7972202655166787,
                                           -0.90018282739798461, 3.3401033852664566, 0.0,
                                           0.88105841856465605,  0.43911718823267476};
      std::vector<Real> const scalarFlux{0.61647714248450702, 1.2410979509359938, 1.8463805541782863};
      Real thermalEnergyFlux = 0.9702629326292449;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(rightRotationRightSide, rightRotationLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of right fast shock\n"
          "Right State: Right of right fast shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.48777637414577313,  2.3709438477809708, -1.7282900552525988,
                                           -0.86414423547773778, 2.8885015704245069, 0.0,
                                           0.77133731061645838,  0.38566794697432505};
      std::vector<Real> const scalarFlux{0.53996724117661621, 1.0870674521621893, 1.6172294888076189};
      Real thermalEnergyFlux = 0.84330016382608752;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(rightFastShockLeftSide, rightFastShockRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of right fast shock\n"
          "Right State: Left of right fast shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.040639426423817904, 1.0717156491947966,  -1.2612066401572222,
                                           -0.63060225433149875, 0.15803727234007203, 0.0,
                                           0.042555541396817498, 0.021277678888288909};
      std::vector<Real> const scalarFlux{0.044987744655527385, 0.090569777630660403, 0.13474059488003065};
      Real thermalEnergyFlux = 0.060961577855018087;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(rightFastShockRightSide, rightFastShockLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the HLLD Riemann Solver using various states and waves from
 * the Ryu & Jones 4d Shock tube
 *
 */
TEST_F(tMHDCalculateHLLDFluxesCUDA, RyuAndJones4dShockTubeCorrectInputExpectCorrectOutput)
{
  // Constant Values
  Real const gamma = 5. / 3.;
  Real const Bx    = 0.7;
  std::vector<Real> const primitiveScalar{1.1069975296, 2.2286185018, 3.3155141875};

  // States
  std::vector<Real> const  // | Density | X-Velocity | Y-Velocity |  Z-Velocity |
                           // Pressure | X-Magnetic Field | Y-Magnetic Field |
                           // Z-Magnetic Field | Adiabatic Index | Passive Scalars |
      leftICs           = Primitive_2_Conserved({1.0, 0.0, 0.0, 0.0, 1.0, Bx, 0.0, 0.0}, gamma, primitiveScalar),
      hydroRareLeftSide = Primitive_2_Conserved(
          {0.990414, 0.012415, 1.458910e-58, 6.294360e-59, 0.984076, Bx, 1.252355e-57, 5.366795e-58}, gamma,
          primitiveScalar),
      hydroRareRightSide = Primitive_2_Conserved(
          {0.939477, 0.079800, 1.557120e-41, 7.505190e-42, 0.901182, Bx, 1.823624e-40, 8.712177e-41}, gamma,
          primitiveScalar),
      switchOnSlowShockLeftSide = Primitive_2_Conserved(
          {0.939863, 0.079142, 1.415730e-02, 7.134030e-03, 0.901820, Bx, 2.519650e-02, 1.290082e-02}, gamma,
          primitiveScalar),
      switchOnSlowShockRightSide = Primitive_2_Conserved(
          {0.651753, 0.322362, 8.070540e-01, 4.425110e-01, 0.490103, Bx, 6.598380e-01, 3.618000e-01}, gamma,
          primitiveScalar),
      contactLeftSide = Primitive_2_Conserved(
          {0.648553, 0.322525, 8.072970e-01, 4.426950e-01, 0.489951, Bx, 6.599295e-01, 3.618910e-01}, gamma,
          primitiveScalar),
      contactRightSide = Primitive_2_Conserved(
          {0.489933, 0.322518, 8.073090e-01, 4.426960e-01, 0.489980, Bx, 6.599195e-01, 3.618850e-01}, gamma,
          primitiveScalar),
      slowShockLeftSide = Primitive_2_Conserved(
          {0.496478, 0.308418, 8.060830e-01, 4.420150e-01, 0.489823, Bx, 6.686695e-01, 3.666915e-01}, gamma,
          primitiveScalar),
      slowShockRightSide = Primitive_2_Conserved(
          {0.298260, -0.016740, 2.372870e-01, 1.287780e-01, 0.198864, Bx, 8.662095e-01, 4.757390e-01}, gamma,
          primitiveScalar),
      rotationLeftSide = Primitive_2_Conserved(
          {0.298001, -0.017358, 2.364790e-01, 1.278540e-01, 0.198448, Bx, 8.669425e-01, 4.750845e-01}, gamma,
          primitiveScalar),
      rotationRightSide = Primitive_2_Conserved(
          {0.297673, -0.018657, 1.059540e-02, 9.996860e-01, 0.197421, Bx, 9.891580e-01, 1.024949e-04}, gamma,
          primitiveScalar),
      fastRareLeftSide = Primitive_2_Conserved(
          {0.297504, -0.020018, 1.137420e-02, 1.000000e+00, 0.197234, Bx, 9.883860e-01, -4.981931e-17}, gamma,
          primitiveScalar),
      fastRareRightSide = Primitive_2_Conserved(
          {0.299996, -0.000033, 1.855120e-05, 1.000000e+00, 0.199995, Bx, 9.999865e-01, 1.737190e-16}, gamma,
          primitiveScalar),
      rightICs = Primitive_2_Conserved({0.3, 0.0, 0.0, 1.0, 0.2, Bx, 1.0, 0.0}, gamma, primitiveScalar);

  for (size_t direction = 0; direction < 3; direction++) {
    // Initial Condition Checks
    {
      std::string const outputString{
          "Left State:  Left Ryu & Jones 4d state\n"
          "Right State: Left Ryu & Jones 4d state\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0, 0.75499999999999989, 0, 0, 2.2204460492503131e-16, 0.0, 0, 0};
      std::vector<Real> const scalarFlux{0, 0, 0};
      Real thermalEnergyFlux             = 0.0;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftICs, leftICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Ryu & Jones 4d state\n"
          "Right State: Right Ryu & Jones 4d state\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          -5.5511151231257827e-17, 0.45500000000000013, -0.69999999999999996, -5.5511151231257827e-17, 0, 0.0, 0,
          -0.69999999999999996};
      std::vector<Real> const scalarFlux{-6.1450707278254418e-17, -1.2371317869019906e-16, -1.8404800947169341e-16};
      Real thermalEnergyFlux             = 0.0;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightICs, rightICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left Ryu & Jones 4d state\n"
          "Right State: Right Ryu & Jones 4d state\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.092428729855986602, 0.53311593977445149,  -0.39622049648437296,
                                           -0.21566989083797167, -0.13287876964320211, 0.0,
                                           -0.40407579574102892, -0.21994567048141428};
      std::vector<Real> const scalarFlux{0.10231837561464294, 0.20598837745492582, 0.30644876517012837};
      Real thermalEnergyFlux             = 0.13864309478397996;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftICs, rightICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Ryu & Jones 4d state\n"
          "Right State: Left Ryu & Jones 4d state\n"
          "HLLD State: Right Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-0.092428729855986602, 0.53311593977445149, -0.39622049648437296,
                                           0.21566989083797167,   0.13287876964320211, 0.0,
                                           0.40407579574102892,   -0.21994567048141428};
      std::vector<Real> const scalarFlux{-0.10231837561464294, -0.20598837745492582, -0.30644876517012837};
      Real thermalEnergyFlux             = -0.13864309478397996;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightICs, leftICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }

    // Cross wave checks
    {
      std::string const outputString{
          "Left State:  Left side of pure hydrodynamic rarefaction\n"
          "Right State: Right side of pure hydrodynamic rarefaction\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.074035256375659553,    0.66054553664209648,    -6.1597070943493028e-41,
                                           -2.9447391900433873e-41, 0.1776649658235645,     0.0,
                                           -6.3466063324344113e-41, -3.0340891384335242e-41};
      std::vector<Real> const scalarFlux{0.081956845911157775, 0.16499634214430131, 0.24546494288869905};
      Real thermalEnergyFlux             = 0.11034221894046368;
      std::vector<Real> const testFluxes = Compute_Fluxes(hydroRareLeftSide, hydroRareRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right side of pure hydrodynamic rarefaction\n"
          "Right State: Left side of pure hydrodynamic rarefaction\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.013336890338886076,    0.74071279157971992,   -6.1745213352160876e-41,
                                           -2.9474651270630147e-41, 0.033152482405470307,  0.0,
                                           6.2022392844946449e-41,  2.9606965476795895e-41};
      std::vector<Real> const scalarFlux{0.014763904657692993, 0.029722840565719184, 0.044218649135708464};
      Real thermalEnergyFlux             = 0.019189877201961154;
      std::vector<Real> const testFluxes = Compute_Fluxes(hydroRareRightSide, hydroRareLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of switch on slow shock\n"
          "Right State: Right of switch on slow shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.19734622040826083,  0.47855039640569758, -0.3392293209655618,
                                           -0.18588204716255491, 0.10695446263054809, 0.0,
                                           -0.3558357543098733,  -0.19525093130352045};
      std::vector<Real> const scalarFlux{0.21846177846784187, 0.43980943806215089, 0.65430419361309078};
      Real thermalEnergyFlux = 0.2840373040888583;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(switchOnSlowShockLeftSide, switchOnSlowShockRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of switch on slow shock\n"
          "Right State: Left of switch on slow shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.097593254768855386,  0.76483698872352757,    -0.02036438492698419,
                                           -0.010747481940703562, 0.25327551496496836,    0.0,
                                           -0.002520109973016129, -0.00088262199017708799};
      std::vector<Real> const scalarFlux{0.10803549193474633, 0.21749813322875222, 0.32357182079044206};
      Real thermalEnergyFlux = 0.1100817647375162;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(switchOnSlowShockRightSide, switchOnSlowShockLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of contact discontinuity\n"
          "Right State: Right of contact discontinuity\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.2091677440314007,   0.5956612619664029,  -0.29309091669513981,
                                           -0.16072556008504282, 0.19220050968424285, 0.0,
                                           -0.35226977371803297, -0.19316940226499904};
      std::vector<Real> const scalarFlux{0.23154817591476573, 0.46615510432814616, 0.69349862290347741};
      Real thermalEnergyFlux             = 0.23702444986592192;
      std::vector<Real> const testFluxes = Compute_Fluxes(contactLeftSide, contactRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of contact discontinuity\n"
          "Right State: Left of contact discontinuity\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.15801775068597168,  0.57916072367837657, -0.33437339604094024,
                                           -0.18336617461176744, 0.16789791355547545, 0.0,
                                           -0.3522739911439669,  -0.19317084712861482};
      std::vector<Real> const scalarFlux{0.17492525964231936, 0.35216128279157616, 0.52391009427617696};
      Real thermalEnergyFlux             = 0.23704936434506069;
      std::vector<Real> const testFluxes = Compute_Fluxes(contactRightSide, contactLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of slow shock\n"
          "Right State: Right of slow shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.11744487326715558,  0.66868230621718128,  -0.35832022960458892,
                                           -0.19650694834641164, 0.057880816021092185, 0.0,
                                           -0.37198011453582402, -0.20397277844271294};
      std::vector<Real> const scalarFlux{0.13001118457092631, 0.26173981750473918, 0.38939014356639379};
      Real thermalEnergyFlux             = 0.1738058891582446;
      std::vector<Real> const testFluxes = Compute_Fluxes(slowShockLeftSide, slowShockRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of slow shock\n"
          "Right State: Left of slow shock\n"
          "HLLD State: Left Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0.038440990187426027, 0.33776683678923869,  -0.62583241538732792,
                                           -0.3437911783906169,  -0.13471828103488348, 0.0,
                                           -0.15165427985881363, -0.082233932588833825};
      std::vector<Real> const scalarFlux{0.042554081172858457, 0.085670301959209896, 0.12745164834795927};
      Real thermalEnergyFlux             = 0.038445630017261548;
      std::vector<Real> const testFluxes = Compute_Fluxes(slowShockRightSide, slowShockLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of rotation/Alfven wave\n"
          "Right State: Right of rotation/Alfven wave\n"
          "HLLD State: Right Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-0.0052668366104996478, 0.44242247672452317,  -0.60785196341731951,
                                           -0.33352435102145184,   -0.21197843894720192, 0.0,
                                           -0.18030635192654354,   -0.098381113757603278};
      std::vector<Real> const scalarFlux{-0.0058303751166299484, -0.011737769516117116, -0.017462271505355991};
      Real thermalEnergyFlux             = -0.0052395622905745485;
      std::vector<Real> const testFluxes = Compute_Fluxes(rotationLeftSide, rotationRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of rotation/Alfven wave\n"
          "Right State: Left of rotation/Alfven wave\n"
          "HLLD State: Right Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-0.005459628948343731,  0.4415038084184626,    -0.69273580053867279,
                                           -0.0051834737482743809, -0.037389286119015486, 0.0,
                                           -0.026148289294373184,  -0.69914753968916865};
      std::vector<Real> const scalarFlux{-0.0060437957583491572, -0.012167430087241717, -0.018101477236719343};
      Real thermalEnergyFlux             = -0.0054536013916442853;
      std::vector<Real> const testFluxes = Compute_Fluxes(rotationRightSide, rotationLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left of fast rarefaction\n"
          "Right State: Right of fast rarefaction\n"
          "HLLD State: Right Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-0.0059354802028144249, 0.44075681881443612,   -0.69194176811725872,
                                           -0.0059354802028144804, -0.040194357552219451, 0.0,
                                           -0.027710302430178135,  -0.70000000000000007};
      std::vector<Real> const scalarFlux{-0.0065705619215052757, -0.013227920997059845, -0.019679168822056604};
      Real thermalEnergyFlux             = -0.0059354109546219782;
      std::vector<Real> const testFluxes = Compute_Fluxes(fastRareLeftSide, fastRareRightSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right of fast rarefaction\n"
          "Right State: Left of fast rarefaction\n"
          "HLLD State: Right Double Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-3.0171858819483255e-05, 0.45503057873272706,     -0.69998654276213712,
                                           -3.0171858819427744e-05, -0.00014827469339251387, 0.0,
                                           -8.2898844654399895e-05, -0.69999999999999984};
      std::vector<Real> const scalarFlux{-3.340017317660794e-05, -6.7241562798797897e-05, -0.00010003522597924373};
      Real thermalEnergyFlux             = -3.000421709818028e-05;
      std::vector<Real> const testFluxes = Compute_Fluxes(fastRareRightSide, fastRareLeftSide, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the HLLD Riemann Solver using various states and waves from
 * the Einfeldt Strong Rarefaction (EFR)
 *
 */
TEST_F(tMHDCalculateHLLDFluxesCUDA, EinfeldtStrongRarefactionCorrectInputExpectCorrectOutput)
{
  // Constant Values
  Real const gamma = 5. / 3.;
  Real const V0    = 2.;
  Real const Vy    = 0.0;
  Real const Vz    = 0.0;
  Real const Bx    = 0.0;
  Real const Bz    = 0.0;

  std::vector<Real> const primitiveScalar{1.1069975296, 2.2286185018, 3.3155141875};

  // States
  std::vector<Real> const  // | Density | X-Velocity | Y-Velocity | Z-Velocity |
                           // Pressure | X-Magnetic Field | Y-Magnetic Field |
                           // Z-Magnetic Field | Adiabatic Index | Passive Scalars |
      leftICs = Primitive_2_Conserved({1.0, -V0, Vy, Vz, 0.45, Bx, 0.5, Bz}, gamma, primitiveScalar),
      leftRarefactionCenter =
          Primitive_2_Conserved({0.368580, -1.180830, Vy, Vz, 0.111253, Bx, 0.183044, Bz}, gamma, primitiveScalar),
      leftVxTurnOver =
          Primitive_2_Conserved({0.058814, -0.125475, Vy, Vz, 0.008819, Bx, 0.029215, Bz}, gamma, primitiveScalar),
      midPoint =
          Primitive_2_Conserved({0.034658, 0.000778, Vy, Vz, 0.006776, Bx, 0.017333, Bz}, gamma, primitiveScalar),
      rightVxTurnOver =
          Primitive_2_Conserved({0.062587, 0.152160, Vy, Vz, 0.009521, Bx, 0.031576, Bz}, gamma, primitiveScalar),
      rightRarefactionCenter =
          Primitive_2_Conserved({0.316485, 1.073560, Vy, Vz, 0.089875, Bx, 0.159366, Bz}, gamma, primitiveScalar),
      rightICs = Primitive_2_Conserved({1.0, V0, Vy, Vz, 0.45, Bx, 0.5, Bz}, gamma, primitiveScalar);

  for (size_t direction = 0; direction < 3; direction++) {
    // Initial Condition Checks
    {
      std::string const outputString{
          "Left State:  Left Einfeldt Strong Rarefaction state\n"
          "Right State: Left Einfeldt Strong Rarefaction state\n"
          "HLLD State: Right"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-2, 4.5750000000000002, -0, -0, -6.75, 0.0, -1, -0};
      std::vector<Real> const scalarFlux{-2.2139950592000002, -4.4572370036000004, -6.6310283749999996};
      Real thermalEnergyFlux             = -1.3499999999999996;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftICs, leftICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Einfeldt Strong Rarefaction state\n"
          "Right State: Right Einfeldt Strong Rarefaction state\n"
          "HLLD State: Left"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{2, 4.5750000000000002, 0, 0, 6.75, 0.0, 1, 0};
      std::vector<Real> const scalarFlux{2.2139950592000002, 4.4572370036000004, 6.6310283749999996};
      Real thermalEnergyFlux             = 1.3499999999999996;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightICs, rightICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left Einfeldt Strong Rarefaction state\n"
          "Right State: Right Einfeldt Strong Rarefaction state\n"
          "HLLD State: Left Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0, -1.4249999999999998, -0, -0, 0, 0.0, 0, -0};
      std::vector<Real> const scalarFlux{0, 0, 0};
      Real thermalEnergyFlux             = 0.0;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftICs, rightICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Einfeldt Strong Rarefaction state\n"
          "Right State: Left Einfeldt Strong Rarefaction state\n"
          "HLLD State: Left Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0, 10.574999999999999, 0, 0, 0, 0.0, 0, 0};
      std::vector<Real> const scalarFlux{0, 0, 0};
      Real thermalEnergyFlux             = 0.0;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightICs, leftICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }

    // Intermediate state checks
    {
      std::string const outputString{
          "Left State:  Left Einfeldt Strong Rarefaction state\n"
          "Right State: Left rarefaction center\n"
          "HLLD State: Right"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          -0.43523032140000006, 0.64193857338676208, -0, -0, -0.67142479846795033, 0.0, -0.21614384652000002, -0};
      std::vector<Real> const scalarFlux{-0.48179889059681413, -0.9699623468164007, -1.4430123054318851};
      Real thermalEnergyFlux             = -0.19705631998499995;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftICs, leftRarefactionCenter, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left rarefaction center\n"
          "Right State: Left Einfeldt Strong Rarefaction state\n"
          "HLLD State: Right"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-2, 4.5750000000000002, -0, -0, -6.75, 0.0, -1, -0};
      std::vector<Real> const scalarFlux{-2.2139950592000002, -4.4572370036000004, -6.6310283749999996};
      Real thermalEnergyFlux             = -1.3499999999999996;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftRarefactionCenter, leftICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left rarefaction center\n"
          "Right State: Left Vx turnover point\n"
          "HLLD State: Right Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          -0.023176056428381629, -2.0437812714100764e-05, 0, 0, -0.00098843768795337005, 0.0, -0.011512369309265979, 0};
      std::vector<Real> const scalarFlux{-0.025655837212088663, -0.051650588155052128, -0.076840543898599858};
      Real thermalEnergyFlux             = -0.0052127803322822184;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftRarefactionCenter, leftVxTurnOver, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left Vx turnover point\n"
          "Right State: Left rarefaction center\n"
          "HLLD State: Right Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          -0.43613091609689758, 0.64135749005731213, 0, 0, -0.67086080671260462, 0.0, -0.21659109937066717, 0};
      std::vector<Real> const scalarFlux{-0.48279584670145054, -0.9719694288205295, -1.445998239926636};
      Real thermalEnergyFlux             = -0.19746407621898149;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftVxTurnOver, leftRarefactionCenter, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Left Vx turnover point\n"
          "Right State: Midpoint\n"
          "HLLD State: Right Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          -0.0011656375857387598, 0.0062355370788444902, 0, 0, -0.00055517615333601446, 0.0, -0.0005829533231464588, 0};
      std::vector<Real> const scalarFlux{-0.0012903579278217153, -0.0025977614899708843, -0.0038646879530001054};
      Real thermalEnergyFlux             = -0.00034184143405415065;
      std::vector<Real> const testFluxes = Compute_Fluxes(leftVxTurnOver, midPoint, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Midpoint\n"
          "Right State: Left Vx turnover point\n"
          "HLLD State: Right Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          -0.0068097924351817191, 0.010501781004354172, 0, 0, -0.0027509360975397175, 0.0, -0.0033826654536986789, 0};
      std::vector<Real> const scalarFlux{-0.0075384234028349319, -0.015176429414463658, -0.022577963432775162};
      Real thermalEnergyFlux             = -0.001531664896602873;
      std::vector<Real> const testFluxes = Compute_Fluxes(midPoint, leftVxTurnOver, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Midpoint\n"
          "Right State: Right Vx turnover point\n"
          "HLLD State: Left Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          0.0013952100758668729, 0.0061359407125797273, 0, 0, 0.00065984543596031629, 0.0, 0.00069776606396793105, 0};
      std::vector<Real> const scalarFlux{0.001544494107257657, 0.0031093909889746947, 0.0046258388010795683};
      Real thermalEnergyFlux             = 0.00040916715364737997;
      std::vector<Real> const testFluxes = Compute_Fluxes(midPoint, rightVxTurnOver, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Vx turnover point\n"
          "Right State: Midpoint\n"
          "HLLD State: Left Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          0.0090024688079190333, 0.011769373146023688, 0, 0, 0.003725251767222792, 0.0, 0.0045418689996141555, 0};
      std::vector<Real> const scalarFlux{0.0099657107306674268, 0.020063068547205749, 0.029847813055181766};
      Real thermalEnergyFlux             = 0.0020542406295284269;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightVxTurnOver, midPoint, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Vx turnover point\n"
          "Right State: Right rarefaction center\n"
          "HLLD State: Left Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          0.023310393229073981, 0.0033086897645311728, 0, 0, 0.0034208520409618887, 0.0, 0.011760413130542123, 0};
      std::vector<Real> const scalarFlux{0.025804547718589466, 0.051949973634547723, 0.077285939467198722};
      Real thermalEnergyFlux             = 0.0053191138878843835;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightVxTurnOver, rightRarefactionCenter, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right rarefaction center\n"
          "Right State: Right Vx turnover point\n"
          "HLLD State: Left Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          0.33914253809565298, 0.46770133685446141, 0, 0, 0.46453338019960133, 0.0, 0.17077520175095764, 0};
      std::vector<Real> const scalarFlux{0.37542995185416178, 0.75581933514738364, 1.1244318966408966};
      Real thermalEnergyFlux             = 0.1444638874418068;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightRarefactionCenter, rightVxTurnOver, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right rarefaction center\n"
          "Right State: Right Einfeldt Strong Rarefaction state\n"
          "HLLD State: Left"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          0.33976563660000003, 0.46733255780629601, 0, 0, 0.46427650313257612, 0.0, 0.17108896296000001, 0};
      std::vector<Real> const scalarFlux{0.37611972035917141, 0.75720798400261535, 1.1264977885722693};
      Real thermalEnergyFlux             = 0.14472930749999999;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightRarefactionCenter, rightICs, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Right Einfeldt Strong Rarefaction state\n"
          "Right State: Right rarefaction center\n"
          "HLLD State: Left"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{2, 4.5750000000000002, 0, 0, 6.75, 0.0, 1, 0};
      std::vector<Real> const scalarFlux{2.2139950592000002, 4.4572370036000004, 6.6310283749999996};
      Real thermalEnergyFlux             = 1.3499999999999996;
      std::vector<Real> const testFluxes = Compute_Fluxes(rightICs, rightRarefactionCenter, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the HLLD Riemann Solver using the constant states from the
 * examples in cholla/examples/3D
 *
 */
TEST_F(tMHDCalculateHLLDFluxesCUDA, ConstantStatesExpectCorrectFlux)
{
  // Constant Values
  Real const gamma = 5. / 3.;

  std::vector<Real> const primitiveScalar{1.1069975296, 2.2286185018, 3.3155141875};

  // States
  std::vector<Real> const  // | Density | X-Velocity | Y-Velocity | Z-Velocity |
                           // Pressure    | X-Magnetic Field | Y-Magnetic Field |
                           // Z-Magnetic Field | Adiabatic Index | Passive Scalars |
      zeroMagneticField =
          Primitive_2_Conserved({1e4, 0.0, 0.0, 0.0, 1.380658E-5, 0.0, 0.0, 0.0}, gamma, primitiveScalar),
      onesMagneticField =
          Primitive_2_Conserved({1e4, 0.0, 0.0, 0.0, 1.380658E-5, 1.0, 1.0, 1.0}, gamma, primitiveScalar);

  for (size_t direction = 0; direction < 3; direction++) {
    {
      std::string const outputString{
          "Left State:  Constant state, zero magnetic field\n"
          "Right State: Constant state, zero magnetic field\n"
          "HLLD State: Left Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{0, 1.380658e-05, 0, 0, 0, 0, 0, 0};
      std::vector<Real> const scalarFlux{0, 0, 0};
      Real thermalEnergyFlux             = 0.;
      std::vector<Real> const testFluxes = Compute_Fluxes(zeroMagneticField, zeroMagneticField, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Constant state, ones magnetic field\n"
          "Right State: Constant state, ones magnetic field\n"
          "HLLD State: Left Double Star"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{
          -1.42108547152020037174e-14, 0.50001380657999994,   -1, -1, -1.7347234759768071e-18, 0.0,
          3.4694469519536142e-18,      3.4694469519536142e-18};
      std::vector<Real> const scalarFlux{1.5731381063233131e-14, 3.1670573744690958e-14, 4.7116290424753513e-14};
      Real thermalEnergyFlux             = 0.;
      std::vector<Real> const testFluxes = Compute_Fluxes(onesMagneticField, onesMagneticField, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the HLLD Riemann Solver with the degenerate state
 *
 */
TEST_F(tMHDCalculateHLLDFluxesCUDA, DegenerateStateCorrectInputExpectCorrectOutput)
{
  // Constant Values
  Real const gamma = 5. / 3.;
  std::vector<Real> const primitiveScalar{1.1069975296, 2.2286185018, 3.3155141875};

  // State
  std::vector<Real> const  // | Density | X-Velocity | Y-Velocity | Z-Velocity |
                           // Pressure | X-Magnetic Field | Y-Magnetic Field |
                           // Z-Magnetic Field | Adiabatic Index | Passive
                           // Scalars |
      state = Primitive_2_Conserved({1.0, 1.0, 1.0, 1.0, 1.0, 3.0E4, 1.0, 1.0}, gamma, primitiveScalar);

  std::vector<Real> const fiducialFlux{1, -449999997, -29999, -29999, -59994, 0.0, -29999, -29999};
  std::vector<Real> const scalarFlux{1.1069975296000001, 2.2286185018000002, 3.3155141874999998};
  Real thermalEnergyFlux = 1.5;
  std::string const outputString{
      "Left State:  Degenerate state\n"
      "Right State: Degenerate state\n"
      "HLLD State: Left Double Star State"};

  // Compute the fluxes and check for correctness
  // Order of Fluxes is rho, vec(V), E, vec(B)
  // If you run into issues with the energy try 0.001953125 instead.
  // That's what I got when running the Athena solver on its own. Running
  // the Athena solver with theses tests gave me -0.00080700946455175148
  // though
  for (size_t direction = 0; direction < 3; direction++) {
    std::vector<Real> const testFluxes = Compute_Fluxes(state, state, gamma, direction);
    Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the HLLD Riemann Solver with all zeroes
 *
 */
TEST_F(tMHDCalculateHLLDFluxesCUDA, AllZeroesExpectAllZeroes)
{
  // Constant Values
  Real const gamma = 5. / 3.;

  // State
  size_t numElements = 8;
    #ifdef SCALAR
  numElements += 3;
    #endif  // SCALAR

  std::vector<Real> const state(numElements, 0.0);
  std::vector<Real> const fiducialFlux(8, 0.0);
  std::vector<Real> const scalarFlux(3, 0.0);
  Real thermalEnergyFlux = 0.0;

  std::string const outputString{
      "Left State:  All zeroes\n"
      "Right State: All zeroes\n"
      "HLLD State: Right Star State"};

  for (size_t direction = 0; direction < 3; direction++) {
    // Compute the fluxes and check for correctness
    // Order of Fluxes is rho, vec(V), E, vec(B)
    std::vector<Real> const testFluxes = Compute_Fluxes(state, state, gamma, direction);
    Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
  }
}
// =========================================================================

// =========================================================================
/*!
* \brief Test the HLLD Riemann Solver with negative pressure, energy, and
  density.
*
*/
TEST_F(tMHDCalculateHLLDFluxesCUDA, UnphysicalValuesExpectAutomaticFix)
{
  // Constant Values
  Real const gamma = 5. / 3.;

  // States
  std::vector<Real>  // | Density | X-Momentum | Y-Momentum | Z-Momentum |
                     // Energy   | X-Magnetic Field | Y-Magnetic Field |
                     // Z-Magnetic Field | Adiabatic Index | Passive Scalars |
      negativePressure              = {1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0},
      negativeEnergy                = {1.0, 1.0, 1.0, 1.0, -(5 - gamma), 1.0, 1.0, 1.0},
      negativeDensity               = {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
      negativeDensityEnergyPressure = {-1.0, -1.0, -1.0, -1.0, -gamma, 1.0, 1.0, 1.0},
      negativeDensityPressure       = {-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0};

    #ifdef SCALAR
  std::vector<Real> const conservedScalar{1.1069975296, 2.2286185018, 3.3155141875};
  negativePressure.insert(negativePressure.begin() + 5, conservedScalar.begin(),
                          conservedScalar.begin() + grid_enum::nscalars);
  negativeEnergy.insert(negativeEnergy.begin() + 5, conservedScalar.begin(),
                        conservedScalar.begin() + grid_enum::nscalars);
  negativeDensity.insert(negativeDensity.begin() + 5, conservedScalar.begin(),
                         conservedScalar.begin() + grid_enum::nscalars);
  negativeDensityEnergyPressure.insert(negativeDensityEnergyPressure.begin() + 5, conservedScalar.begin(),
                                       conservedScalar.begin() + grid_enum::nscalars);
  negativeDensityPressure.insert(negativeDensityPressure.begin() + 5, conservedScalar.begin(),
                                 conservedScalar.begin() + grid_enum::nscalars);
    #endif  // SCALAR
    #ifdef DE
  negativePressure.push_back(mhd::utils::computeThermalEnergy(
      negativePressure.at(4), negativePressure.at(0), negativePressure.at(1), negativePressure.at(2),
      negativePressure.at(3), negativePressure.at(grid_enum::magnetic_x), negativePressure.at(grid_enum::magnetic_y),
      negativePressure.at(grid_enum::magnetic_z), gamma));
  negativeEnergy.push_back(mhd::utils::computeThermalEnergy(
      negativeEnergy.at(4), negativeEnergy.at(0), negativeEnergy.at(1), negativeEnergy.at(2), negativeEnergy.at(3),
      negativeEnergy.at(grid_enum::magnetic_x), negativeEnergy.at(grid_enum::magnetic_y),
      negativeEnergy.at(grid_enum::magnetic_z), gamma));
  negativeDensity.push_back(mhd::utils::computeThermalEnergy(
      negativeDensity.at(4), negativeDensity.at(0), negativeDensity.at(1), negativeDensity.at(2), negativeDensity.at(3),
      negativeDensity.at(grid_enum::magnetic_x), negativeDensity.at(grid_enum::magnetic_y),
      negativeDensity.at(grid_enum::magnetic_z), gamma));
  negativeDensityEnergyPressure.push_back(mhd::utils::computeThermalEnergy(
      negativeDensityEnergyPressure.at(4), negativeDensityEnergyPressure.at(0), negativeDensityEnergyPressure.at(1),
      negativeDensityEnergyPressure.at(2), negativeDensityEnergyPressure.at(3),
      negativeDensityEnergyPressure.at(grid_enum::magnetic_x), negativeDensityEnergyPressure.at(grid_enum::magnetic_y),
      negativeDensityEnergyPressure.at(grid_enum::magnetic_z), gamma));
  negativeDensityPressure.push_back(mhd::utils::computeThermalEnergy(
      negativeDensityPressure.at(4), negativeDensityPressure.at(0), negativeDensityPressure.at(1),
      negativeDensityPressure.at(2), negativeDensityPressure.at(3), negativeDensityPressure.at(grid_enum::magnetic_x),
      negativeDensityPressure.at(grid_enum::magnetic_y), negativeDensityPressure.at(grid_enum::magnetic_z), gamma));
    #endif  // DE

  for (size_t direction = 0; direction < 3; direction++) {
    {
      std::string const outputString{
          "Left State:  Negative Pressure\n"
          "Right State: Negative Pressure\n"
          "HLLD State: Left Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{1, 1.5, 0, 0, -1.6254793235168146e-16, 0, 0, 0};
      std::vector<Real> const scalarFlux{1.1069975296000001, 2.2286185018000002, 3.3155141874999998};
      Real thermalEnergyFlux             = -1.5;
      std::vector<Real> const testFluxes = Compute_Fluxes(negativePressure, negativePressure, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Negative Energy\n"
          "Right State: Negative Energy\n"
          "HLLD State: Left Star State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{1, 1.5, 0, 0, -1.5, 0, 0, 0};
      std::vector<Real> const scalarFlux{1.1069975296000001, 2.2286185018000002, 3.3155141874999998};
      Real thermalEnergyFlux             = -6.333333333333333;
      std::vector<Real> const testFluxes = Compute_Fluxes(negativeEnergy, negativeEnergy, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Negative Density\n"
          "Right State: Negative Density\n"
          "HLLD State: Left State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{1, 1E+20, 1e+20, 1e+20, -5e+19, 0, 0, 0};
      std::vector<Real> const scalarFlux{1.1069975296000002e+20, 2.2286185018000002e+20, 3.3155141874999997e+20};
      Real thermalEnergyFlux             = -1.5000000000000001e+40;
      std::vector<Real> const testFluxes = Compute_Fluxes(negativeDensity, negativeDensity, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Negative Density, Energy, and Pressure\n"
          "Right State: Negative Density, Energy, and Pressure\n"
          "HLLD State: Right State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{-1, 1E+20, 1E+20, 1E+20, 1.5E+20, 0, 0, 0};
      std::vector<Real> const scalarFlux{-1.1069975296000002e+20, -2.2286185018000002e+20, -3.3155141874999997e+20};
      Real thermalEnergyFlux = 1.5000000000000001e+40;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(negativeDensityEnergyPressure, negativeDensityEnergyPressure, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
    {
      std::string const outputString{
          "Left State:  Negative Density and Pressure\n"
          "Right State: Negative Density and Pressure\n"
          "HLLD State: Left State"};
      // Compute the fluxes and check for correctness
      // Order of Fluxes is rho, vec(V), E, vec(B)
      std::vector<Real> const fiducialFlux{1, 1e+20, 1e+20, 1e+20, -1.5e+20, 0, 0, 0};
      std::vector<Real> const scalarFlux{1.1069975296000002e+20, 2.2286185018000002e+20, 3.3155141874999997e+20};
      Real thermalEnergyFlux = -1.5000000000000001e+40;
      std::vector<Real> const testFluxes =
          Compute_Fluxes(negativeDensityPressure, negativeDensityPressure, gamma, direction);
      Check_Results(fiducialFlux, scalarFlux, thermalEnergyFlux, testFluxes, outputString, direction);
    }
  }
}
// =========================================================================

// =========================================================================
// End of integration tests for the entire HLLD solver. Unit tests are below
// =========================================================================

// =========================================================================
// Unit tests for the contents of the mhd::_internal namespace
// =========================================================================
/*!
 * \brief A struct to hold some basic test values
 *
 */
namespace
{
struct TestParams {
  // List of cases
  std::vector<std::string> names{"Case 1", "Case 2"};

  double const gamma = 5. / 3.;

  std::vector<double> const magneticX{92.75101068883114, 31.588767769990532};

  std::vector<mhd::_internal::State> stateLVec{
      {21.50306776645775, 1.7906564444824999, 0.33040135813215948, 1.500111692877206, 65.751208381099417,
       12.297499156516622, 46.224045698787776, 9.9999999999999995e-21, 5445.3204350339083},
      {48.316634031589935, 0.39291118391272883, 0.69876195899931859, 1.8528943583250035, 38.461354599479826,
       63.744719695704063, 37.703264551707541, 9.9999999999999995e-21, 3241.38784808316}},
      stateRVec{{81.121773176226498, 0.10110493143718589, 0.17103629446142521, 0.41731155351794952, 18.88982523270516,
                 84.991914178754897, 34.852095153095384, 9.9999999999999995e-21, 8605.4286125143772},
                {91.029557388536347, 0.93649399297774782, 0.36277769000180521, 0.095181318599791204, 83.656397841788944,
                 35.910258841630984, 24.052685003977757, 9.9999999999999995e-21, 4491.7524579462979}};

  std::vector<mhd::_internal::StarState> const starStateLVec{
      {28.520995251761526, 1.5746306813243216, 1.3948193325212686, 6.579867455284738, 62.093488291430653,
       62.765890944643196},
      {54.721668215064945, 1.4363926014039052, 1.1515754515491903, 30.450436649083692, 54.279167444036723,
       93.267654555096414}},
      starStateRVec{{49.090695707386047, 1.0519818825796206, 0.68198273634686157, 90.44484278669114, 26.835645069149873,
                     7.4302316959173442},
                    {72.680005044606091, 0.61418047569879897, 0.71813570322922715, 61.33664731346812,
                     98.974446283273181, 10.696380763901459}};

  std::vector<double> totalPressureStar{66.80958736783934, 72.29644038317676};

  std::vector<mhd::_internal::DoubleStarState> const DoubleStarStateVec{
      {0.79104271107837087, 0.97609103551927523, 20.943239839455895, 83.380243826880701, 45.832024557076693,
       std::nan("0")},
      {1.390870320696683, 0.52222643241336986, 83.851481048702098, 80.366712517307832, 55.455301414557297,
       std::nan("0")}};

  std::vector<mhd::_internal::Flux> const flux{
      {12.939239309626116, 65.054814649176265, 73.676928455867824, 16.873647595664387, 52.718887319724693,
       58.989284454159673, 29.976925743532302},
      {81.715245865170729, 56.098850697078028, 2.7172469834037871, 39.701329831928732, 81.63926176158796,
       57.043444592213589, 97.733298271413588}},
      starFlux{{0, 74.90125547448865, 16.989138610622945, 38.541822734846185, 19.095105176247017, 96.239645266242775,
                86.225169282683467},
               {0, 26.812722601652684, 48.349566649914976, 61.228439610525378, 45.432249733131123, 33.053375365947957,
                15.621020824107379}};

  std::vector<mhd::_internal::Speeds> const speed{
      {-22.40376497145191, -19.710500632936679, -0.81760587897407833, 9.6740190040662242, 24.295526347371595},
      {-11.190385012513822, -4.4880642018724357, -0.026643804611559244, 3.4191202933087519, 12.519790189404299}};

  TestParams() = default;
};
}  // namespace
// =========================================================================

// =========================================================================
/*!
 * \brief Test the mhd::_internal::approximateLRWaveSpeeds function
 *
 */
TEST(tMHDHlldInternalApproximateLRWaveSpeeds, CorrectInputExpectCorrectOutput)
{
  TestParams const parameters;
  std::vector<double> const fiducialSpeedL{-22.40376497145191, -11.190385012513822};
  std::vector<double> const fiducialSpeedR{24.295526347371595, 12.519790189404299};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    mhd::_internal::Speeds testSpeed = mhd::_internal::approximateLRWaveSpeeds(
        parameters.stateLVec.at(i), parameters.stateRVec.at(i), parameters.magneticX.at(i), parameters.gamma);

    // Now check results
    testing_utilities::Check_Results(fiducialSpeedL[i], testSpeed.L, parameters.names.at(i) + ", SpeedL");
    testing_utilities::Check_Results(fiducialSpeedR.at(i), testSpeed.R, parameters.names.at(i) + ", SpeedR");
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the mhd::_internal::approximateMiddleWaveSpeed function
 *
 */
TEST(tMHDHlldInternalApproximateMiddleWaveSpeed, CorrectInputExpectCorrectOutput)
{
  TestParams const parameters;

  std::vector<double> const fiducialSpeedM{-0.81760587897407833, -0.026643804611559244};

  mhd::_internal::Speeds testSpeed;

  for (size_t i = 0; i < parameters.names.size(); i++) {
    testSpeed.M = mhd::_internal::approximateMiddleWaveSpeed(parameters.stateLVec.at(i), parameters.stateRVec.at(i),
                                                             parameters.speed.at(i));

    // Now check results
    testing_utilities::Check_Results(fiducialSpeedM.at(i), testSpeed.M, parameters.names.at(i) + ", SpeedM");
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the mhd::_internal::approximateStarWaveSpeed function
 *
 */
TEST(tMHDHlldInternalApproximateStarWaveSpeed, CorrectInputExpectCorrectOutput)
{
  TestParams const parameters;
  std::vector<double> const fiducialSpeedStarL{-18.18506608966894, -4.2968910457518161};
  std::vector<double> const fiducialSpeedStarR{12.420292938368167, 3.6786718447209252};

  mhd::_internal::Speeds testSpeed;

  for (size_t i = 0; i < parameters.names.size(); i++) {
    testSpeed.LStar = mhd::_internal::approximateStarWaveSpeed(parameters.starStateLVec.at(i), parameters.speed.at(i),
                                                               parameters.magneticX.at(i), -1);
    testSpeed.RStar = mhd::_internal::approximateStarWaveSpeed(parameters.starStateRVec.at(i), parameters.speed.at(i),
                                                               parameters.magneticX.at(i), 1);

    // Now check results
    testing_utilities::Check_Results(fiducialSpeedStarL.at(i), testSpeed.LStar,
                                     parameters.names.at(i) + ", SpeedStarL");
    testing_utilities::Check_Results(fiducialSpeedStarR.at(i), testSpeed.RStar,
                                     parameters.names.at(i) + ", SpeedStarR");
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the mhd::_internal::_nonStarFluxes function
 *
 */
TEST(tMHDHlldInternalNonStarFluxes, CorrectInputExpectCorrectOutput)
{
  TestParams const parameters;

  std::vector<mhd::_internal::Flux> fiducialFlux{
      {38.504606872151484, -3088.4810263278778, -1127.8835013070616, -4229.5657456907293, -12344.460641662206,
       -8.6244637840856555, -56.365490339906408},
      {18.984145880030045, 2250.9966820900618, -2000.3517480656785, -1155.8240512956793, -2717.2127176227905,
       2.9729840344910059, -43.716615275067923}};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    mhd::_internal::Flux testFlux =
        mhd::_internal::nonStarFluxes(parameters.stateLVec.at(i), parameters.magneticX.at(i));

    // Now check results
    testing_utilities::Check_Results(fiducialFlux[i].density, testFlux.density,
                                     parameters.names.at(i) + ", DensityFlux");
    testing_utilities::Check_Results(fiducialFlux[i].momentumX, testFlux.momentumX,
                                     parameters.names.at(i) + ", MomentumFluxX");
    testing_utilities::Check_Results(fiducialFlux[i].momentumY, testFlux.momentumY,
                                     parameters.names.at(i) + ", MomentumFluxY");
    testing_utilities::Check_Results(fiducialFlux[i].momentumZ, testFlux.momentumZ,
                                     parameters.names.at(i) + ", MomentumFluxZ");
    testing_utilities::Check_Results(fiducialFlux[i].magneticY, testFlux.magneticY,
                                     parameters.names.at(i) + ", MagneticFluxY");
    testing_utilities::Check_Results(fiducialFlux[i].magneticZ, testFlux.magneticZ,
                                     parameters.names.at(i) + ", MagneticFluxZ");
    testing_utilities::Check_Results(fiducialFlux[i].energy, testFlux.energy, parameters.names.at(i) + ", EnergyFlux");
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the mhd::_internal::computeStarState function in the
 * non-degenerate case
 *
 */
TEST(tMHDHlldInternalComputeStarState, CorrectInputNonDegenerateExpectCorrectOutput)
{
  TestParams const parameters;

  std::vector<mhd::_internal::StarState> fiducialStarState{
      {24.101290139122913, 1.4626377138501221, 5.7559806612277464, 1023.8840191068900, 18.648382121236992,
       70.095850905078336},
      {50.132466596958501, 0.85967712862308099, 1.9480712959548112, 172.06840532772659, 66.595692901872582,
       39.389537509454122}};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    mhd::_internal::StarState testStarState =
        mhd::_internal::computeStarState(parameters.stateLVec.at(i), parameters.speed.at(i), parameters.speed.at(i).L,
                                         parameters.magneticX.at(i), parameters.totalPressureStar.at(i));

    // Now check results
    testing_utilities::Check_Results(fiducialStarState.at(i).velocityY, testStarState.velocityY,
                                     parameters.names.at(i) + ", VelocityStarY");
    testing_utilities::Check_Results(fiducialStarState.at(i).velocityZ, testStarState.velocityZ,
                                     parameters.names.at(i) + ", VelocityStarZ");
    testing_utilities::Check_Results(fiducialStarState.at(i).energy, testStarState.energy,
                                     parameters.names.at(i) + ", EnergyStar");
    testing_utilities::Check_Results(fiducialStarState.at(i).magneticY, testStarState.magneticY,
                                     parameters.names.at(i) + ", MagneticStarY");
    testing_utilities::Check_Results(fiducialStarState.at(i).magneticZ, testStarState.magneticZ,
                                     parameters.names.at(i) + ", MagneticStarZ");
  }
}

/*!
 * \brief Test the mhd::_internal::starFluxes function in the non-degenerate
 * case
 *
 */
TEST(tMHDHlldInternalStarFluxes, CorrectInputNonDegenerateExpectCorrectOutput)
{
  TestParams const parameters;

  std::vector<mhd::_internal::Flux> fiducialFlux{
      {-45.270724071132321, 1369.1771532285088, -556.91765728768155, -2368.4452742393819, -21413.063415617500,
       -83.294404848633300, -504.84138754248409},
      {61.395380340435793, 283.48596932136809, -101.75517013858293, -51.34364892516212, -1413.4750762739586,
       25.139956754826922, 78.863254638038882}};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    mhd::_internal::StarState testStarState =
        mhd::_internal::computeStarState(parameters.stateLVec.at(i), parameters.speed.at(i), parameters.speed.at(i).L,
                                         parameters.magneticX.at(i), parameters.totalPressureStar.at(i));

    mhd::_internal::Flux testFlux =
        mhd::_internal::starFluxes(testStarState, parameters.stateLVec.at(i), parameters.flux.at(i),
                                   parameters.speed.at(i), parameters.speed.at(i).L);

    // Now check results
    testing_utilities::Check_Results(fiducialFlux[i].density, testFlux.density,
                                     parameters.names.at(i) + ", DensityStarFlux");
    testing_utilities::Check_Results(fiducialFlux[i].momentumX, testFlux.momentumX,
                                     parameters.names.at(i) + ", MomentumStarFluxX");
    testing_utilities::Check_Results(fiducialFlux[i].momentumY, testFlux.momentumY,
                                     parameters.names.at(i) + ", MomentumStarFluxY");
    testing_utilities::Check_Results(fiducialFlux[i].momentumZ, testFlux.momentumZ,
                                     parameters.names.at(i) + ", MomentumStarFluxZ");
    testing_utilities::Check_Results(fiducialFlux[i].energy, testFlux.energy,
                                     parameters.names.at(i) + ", EnergyStarFlux");
    testing_utilities::Check_Results(fiducialFlux[i].magneticY, testFlux.magneticY,
                                     parameters.names.at(i) + ", MagneticStarFluxY", 1.0E-13);
    testing_utilities::Check_Results(fiducialFlux[i].magneticZ, testFlux.magneticZ,
                                     parameters.names.at(i) + ", MagneticStarFluxZ", 7.0E-13);
  }
}

/*!
 * \brief Test the mhd::_internal::starFluxes function in the degenerate
 * case
 *
 */
TEST(tMHDHlldInternalComputeStarState, CorrectInputDegenerateExpectCorrectOutput)
{
  TestParams parameters;

  std::vector<mhd::_internal::StarState> fiducialStarState{
      {24.101290139122913, 1.4626377138501221, 5.7559806612277464, 4.5171065808847731e+17, 18.648382121236992,
       70.095850905078336},
      {50.132466596958501, 0.85967712862308099, 1.9480712959548112, 172.06840532772659, 66.595692901872582,
       39.389537509454122}};

  // Used to get us into the degenerate case
  double const totalPressureStarMultiplier = 1E15;
  parameters.stateLVec.at(0).totalPressure *= totalPressureStarMultiplier;

  for (size_t i = 0; i < parameters.names.size(); i++) {
    mhd::_internal::StarState testStarState =
        mhd::_internal::computeStarState(parameters.stateLVec.at(i), parameters.speed.at(i), parameters.speed.at(i).L,
                                         parameters.magneticX.at(i), parameters.totalPressureStar.at(i));

    // Now check results
    testing_utilities::Check_Results(fiducialStarState.at(i).velocityY, testStarState.velocityY,
                                     parameters.names.at(i) + ", VelocityStarY");
    testing_utilities::Check_Results(fiducialStarState.at(i).velocityZ, testStarState.velocityZ,
                                     parameters.names.at(i) + ", VelocityStarZ");
    testing_utilities::Check_Results(fiducialStarState.at(i).energy, testStarState.energy,
                                     parameters.names.at(i) + ", EnergyStar");
    testing_utilities::Check_Results(fiducialStarState.at(i).magneticY, testStarState.magneticY,
                                     parameters.names.at(i) + ", MagneticStarY");
    testing_utilities::Check_Results(fiducialStarState.at(i).magneticZ, testStarState.magneticZ,
                                     parameters.names.at(i) + ", MagneticStarZ");
  }
}

TEST(tMHDHlldInternalStarFluxes, CorrectInputDegenerateExpectCorrectOutput)
{
  TestParams parameters;

  // Used to get us into the degenerate case
  double const totalPressureStarMultiplier = 1E15;

  std::vector<mhd::_internal::Flux> fiducialFlux{
      {-144.2887586578122, 1450.1348804310369, -773.30617492819886, -151.70644305354989, 1378.3797024673304,
       -1056.6283526454272, -340.62268733874163},
      {10.040447333773272, 284.85426012223729, -499.05932057162761, 336.35271628090368, 171.28451793017882,
       162.96661864443826, -524.05361885198215}};

  parameters.totalPressureStar.at(0) *= totalPressureStarMultiplier;
  parameters.totalPressureStar.at(1) *= totalPressureStarMultiplier;

  for (size_t i = 0; i < parameters.names.size(); i++) {
    mhd::_internal::Flux testFlux =
        mhd::_internal::starFluxes(parameters.starStateLVec.at(i), parameters.stateLVec.at(i), parameters.flux.at(i),
                                   parameters.speed.at(i), parameters.speed.at(i).L);

    // Now check results
    testing_utilities::Check_Results(fiducialFlux[i].density, testFlux.density,
                                     parameters.names.at(i) + ", DensityStarFlux");
    testing_utilities::Check_Results(fiducialFlux[i].momentumX, testFlux.momentumX,
                                     parameters.names.at(i) + ", MomentumStarFluxX");
    testing_utilities::Check_Results(fiducialFlux[i].momentumY, testFlux.momentumY,
                                     parameters.names.at(i) + ", MomentumStarFluxY");
    testing_utilities::Check_Results(fiducialFlux[i].momentumZ, testFlux.momentumZ,
                                     parameters.names.at(i) + ", MomentumStarFluxZ");
    testing_utilities::Check_Results(fiducialFlux[i].energy, testFlux.energy,
                                     parameters.names.at(i) + ", EnergyStarFlux");
    testing_utilities::Check_Results(fiducialFlux[i].magneticY, testFlux.magneticY,
                                     parameters.names.at(i) + ", MagneticStarFluxY");
    testing_utilities::Check_Results(fiducialFlux[i].magneticZ, testFlux.magneticZ,
                                     parameters.names.at(i) + ", MagneticStarFluxZ");
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the mhd::_internal::computeDoubleStarState function.
 * Non-degenerate state
 *
 */
TEST(tMHDHlldInternalDoubleStarState, CorrectInputNonDegenerateExpectCorrectOutput)
{
  TestParams const parameters;

  std::vector<mhd::_internal::DoubleStarState> fiducialState{
      {-1.5775383335759607, -3.4914062207842482, 45.259313435283325, 36.670978215630669, -2048.1953674500523,
       1721.0582276783819},
      {3.803188977150934, -4.2662645349592765, 71.787329583230417, 53.189673238238178, -999.79694164635089,
       252.047167522579}};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    mhd::_internal::DoubleStarState const testState = mhd::_internal::computeDoubleStarState(
        parameters.starStateLVec.at(i), parameters.starStateRVec.at(i), parameters.magneticX.at(i),
        parameters.totalPressureStar.at(i), parameters.speed.at(i));

    // Now check results
    testing_utilities::Check_Results(fiducialState.at(i).velocityY, testState.velocityY,
                                     parameters.names.at(i) + ", VelocityDoubleStarY");
    testing_utilities::Check_Results(fiducialState.at(i).velocityZ, testState.velocityZ,
                                     parameters.names.at(i) + ", VelocityDoubleStarZ");
    testing_utilities::Check_Results(fiducialState.at(i).magneticY, testState.magneticY,
                                     parameters.names.at(i) + ", MagneticDoubleStarY");
    testing_utilities::Check_Results(fiducialState.at(i).magneticZ, testState.magneticZ,
                                     parameters.names.at(i) + ", MagneticDoubleStarZ");
    testing_utilities::Check_Results(fiducialState.at(i).energyL, testState.energyL,
                                     parameters.names.at(i) + ", EnergyDoubleStarL");
    testing_utilities::Check_Results(fiducialState.at(i).energyR, testState.energyR,
                                     parameters.names.at(i) + ", EnergyDoubleStarR");
  }
}

/*!
 * \brief Test the mhd::_internal::computeDoubleStarState function in the
 * degenerate state.
 *
 */
TEST(tMHDHlldInternalDoubleStarState, CorrectInputDegenerateExpectCorrectOutput)
{
  TestParams const parameters;

  std::vector<mhd::_internal::DoubleStarState> fiducialState{
      {1.0519818825796206, 0.68198273634686157, 26.835645069149873, 7.4302316959173442, 0.0, 90.44484278669114},
      {0.61418047569879897, 0.71813570322922715, 98.974446283273181, 10.696380763901459, 0.0, 61.33664731346812}};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    mhd::_internal::DoubleStarState const testState =
        mhd::_internal::computeDoubleStarState(parameters.starStateLVec.at(i), parameters.starStateRVec.at(i), 0.0,
                                               parameters.totalPressureStar.at(i), parameters.speed.at(i));

    // Now check results
    testing_utilities::Check_Results(fiducialState.at(i).velocityY, testState.velocityY,
                                     parameters.names.at(i) + ", VelocityDoubleStarY");
    testing_utilities::Check_Results(fiducialState.at(i).velocityZ, testState.velocityZ,
                                     parameters.names.at(i) + ", VelocityDoubleStarZ");
    testing_utilities::Check_Results(fiducialState.at(i).magneticY, testState.magneticY,
                                     parameters.names.at(i) + ", MagneticDoubleStarY");
    testing_utilities::Check_Results(fiducialState.at(i).magneticZ, testState.magneticZ,
                                     parameters.names.at(i) + ", MagneticDoubleStarZ");
    testing_utilities::Check_Results(fiducialState.at(i).energyL, testState.energyL,
                                     parameters.names.at(i) + ", EnergyDoubleStarL");
    testing_utilities::Check_Results(fiducialState.at(i).energyR, testState.energyR,
                                     parameters.names.at(i) + ", EnergyDoubleStarR");
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the mhd::_internal::_doubleStarFluxes function
 *
 */
TEST(tMHDHlldInternalDoubleStarFluxes, CorrectInputExpectCorrectOutput)
{
  TestParams const parameters;

  std::vector<mhd::_internal::Flux> const fiducialFlux{
      {-144.2887586578122, 1450.1348804310369, -332.80193639987715, 83.687152337186944, 604.70003506833029,
       -245.53635448727721, -746.94190287166407},
      {10.040447333773258, 284.85426012223729, -487.87930516727664, 490.91728596722157, 59.061079503595295,
       30.244176588794346, -466.15336272175193}};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    mhd::_internal::Flux const testFlux = mhd::_internal::computeDoubleStarFluxes(
        parameters.DoubleStarStateVec.at(i), parameters.DoubleStarStateVec.at(i).energyL,
        parameters.starStateLVec.at(i), parameters.stateLVec.at(i), parameters.flux.at(i), parameters.speed.at(i),
        parameters.speed.at(i).L, parameters.speed.at(i).LStar);

    // Now check results
    testing_utilities::Check_Results(fiducialFlux[i].density, testFlux.density,
                                     parameters.names.at(i) + ", DensityStarFlux", 5.0E-14);
    testing_utilities::Check_Results(fiducialFlux[i].momentumX, testFlux.momentumX,
                                     parameters.names.at(i) + ", MomentumStarFluxX");
    testing_utilities::Check_Results(fiducialFlux[i].momentumY, testFlux.momentumY,
                                     parameters.names.at(i) + ", MomentumStarFluxY");
    testing_utilities::Check_Results(fiducialFlux[i].momentumZ, testFlux.momentumZ,
                                     parameters.names.at(i) + ", MomentumStarFluxZ");
    testing_utilities::Check_Results(fiducialFlux[i].energy, testFlux.energy,
                                     parameters.names.at(i) + ", EnergyStarFlux");
    testing_utilities::Check_Results(fiducialFlux[i].magneticY, testFlux.magneticY,
                                     parameters.names.at(i) + ", MagneticStarFluxY");
    testing_utilities::Check_Results(fiducialFlux[i].magneticZ, testFlux.magneticZ,
                                     parameters.names.at(i) + ", MagneticStarFluxZ");
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the mhd::_internal::_returnFluxes function
 *
 */
TEST(tMHDHlldInternalReturnFluxes, CorrectInputExpectCorrectOutput)
{
  double const dummyValue = 999;
  mhd::_internal::Flux inputFlux{1, 2, 3, 4, 5, 6, 7};
  mhd::_internal::State inputState{8, 9, 10, 11, 12, 13, 14, 15, 16};

  int threadId = 0;
  int n_cells  = 10;
  int nFields  = 8;  // Total number of conserved fields
    #ifdef SCALAR
  nFields += NSCALARS;
    #endif  // SCALAR
    #ifdef DE
  nFields++;
    #endif  // DE

  // Lambda for finding indices and check if they're correct
  auto findIndex = [](std::vector<double> const &vec, double const &num, int const &fidIndex, std::string const &name) {
    int index = std::distance(vec.begin(), std::find(vec.begin(), vec.end(), num));
    EXPECT_EQ(fidIndex, index) << "Error in " << name << " index" << std::endl;

    return index;
  };

  for (size_t direction = 0; direction < 1; direction++) {
    int o1, o2, o3;
    switch (direction) {
      case 0:
        o1 = 1;
        o2 = 2;
        o3 = 3;
        break;
      case 1:
        o1 = 2;
        o2 = 3;
        o3 = 1;
        break;
      case 2:
        o1 = 3;
        o2 = 1;
        o3 = 2;
        break;
    }

    std::vector<double> testFluxArray(nFields * n_cells, dummyValue);

    // Fiducial Indices
    int const fiducialDensityIndex   = threadId + n_cells * grid_enum::density;
    int const fiducialMomentumIndexX = threadId + n_cells * o1;
    int const fiducialMomentumIndexY = threadId + n_cells * o2;
    int const fiducialMomentumIndexZ = threadId + n_cells * o3;
    int const fiducialEnergyIndex    = threadId + n_cells * grid_enum::Energy;
    int const fiducialMagneticYIndex = threadId + n_cells * (grid_enum::magnetic_x);
    int const fiducialMagneticZIndex = threadId + n_cells * (grid_enum::magnetic_y);

    mhd::_internal::returnFluxes(threadId, o1, o2, o3, n_cells, testFluxArray.data(), inputFlux, inputState);

    // Find the indices for the various fields
    int densityLoc    = findIndex(testFluxArray, inputFlux.density, fiducialDensityIndex, "density");
    int momentumXLocX = findIndex(testFluxArray, inputFlux.momentumX, fiducialMomentumIndexX, "momentum X");
    int momentumYLocY = findIndex(testFluxArray, inputFlux.momentumY, fiducialMomentumIndexY, "momentum Y");
    int momentumZLocZ = findIndex(testFluxArray, inputFlux.momentumZ, fiducialMomentumIndexZ, "momentum Z");
    int energyLoc     = findIndex(testFluxArray, inputFlux.energy, fiducialEnergyIndex, "energy");
    int magneticYLoc  = findIndex(testFluxArray, inputFlux.magneticY, fiducialMagneticYIndex, "magnetic Y");
    int magneticZLoc  = findIndex(testFluxArray, inputFlux.magneticZ, fiducialMagneticZIndex, "magnetic Z");

    for (size_t i = 0; i < testFluxArray.size(); i++) {
      // Skip the already checked indices
      if ((i != densityLoc) and (i != momentumXLocX) and (i != momentumYLocY) and (i != momentumZLocZ) and
          (i != energyLoc) and (i != magneticYLoc) and (i != magneticZLoc)) {
        EXPECT_EQ(dummyValue, testFluxArray.at(i)) << "Unexpected value at index that _returnFluxes shouldn't be "
                                                      "touching"
                                                   << std::endl
                                                   << "Index     = " << i << std::endl
                                                   << "Direction = " << direction << std::endl;
      }
    }
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the mhd::_internal::starTotalPressure function
 *
 */
TEST(tMHDHlldInternalStarTotalPressure, CorrectInputExpectCorrectOutput)
{
  TestParams const parameters;

  std::vector<double> const fiducialPressure{6802.2800807224075, 3476.1984612875144};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real const testPressure = mhd::_internal::starTotalPressure(parameters.stateLVec.at(i), parameters.stateRVec.at(i),
                                                                parameters.speed.at(i));

    // Now check results
    testing_utilities::Check_Results(fiducialPressure.at(i), testPressure,
                                     parameters.names.at(i) + ", total pressure in the star states");
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Test the mhd::_internal::loadState function
 *
 */
TEST(tMHDHlldInternalLoadState, CorrectInputExpectCorrectOutput)
{
  TestParams const parameters;
  int const threadId = 0;
  int const n_cells  = 10;
  std::vector<double> interfaceArray(n_cells * grid_enum::num_fields);
  std::iota(std::begin(interfaceArray), std::end(interfaceArray), 1.);

  std::vector<mhd::_internal::State> const fiducialState{
      {1, 11, 21, 31, 41, 51, 61, 9.9999999999999995e-21, 7462.3749918998346},
      {1, 21, 31, 11, 41, 51, 61, 9.9999999999999995e-21, 7462.3749918998346},
      {1, 31, 11, 21, 41, 51, 61, 9.9999999999999995e-21, 7462.3749918998346},
  };

  for (size_t direction = 0; direction < 3; direction++) {
    int o1, o2, o3;
    switch (direction) {
      case 0:
        o1 = 1;
        o2 = 2;
        o3 = 3;
        break;
      case 1:
        o1 = 2;
        o2 = 3;
        o3 = 1;
        break;
      case 2:
        o1 = 3;
        o2 = 1;
        o3 = 2;
        break;
    }

    mhd::_internal::State const testState = mhd::_internal::loadState(interfaceArray.data(), parameters.magneticX.at(0),
                                                                      parameters.gamma, threadId, n_cells, o1, o2, o3);

    // Now check results
    testing_utilities::Check_Results(fiducialState.at(direction).density, testState.density, ", Density");
    testing_utilities::Check_Results(fiducialState.at(direction).velocityX, testState.velocityX, ", velocityX");
    testing_utilities::Check_Results(fiducialState.at(direction).velocityY, testState.velocityY, ", velocityY");
    testing_utilities::Check_Results(fiducialState.at(direction).velocityZ, testState.velocityZ, ", velocityZ");
    testing_utilities::Check_Results(fiducialState.at(direction).energy, testState.energy, ", energy");
    testing_utilities::Check_Results(fiducialState.at(direction).magneticY, testState.magneticY, ", magneticY");
    testing_utilities::Check_Results(fiducialState.at(direction).magneticZ, testState.magneticZ, ", magneticZ");
    testing_utilities::Check_Results(fiducialState.at(direction).gasPressure, testState.gasPressure, ", gasPressure");
    testing_utilities::Check_Results(fiducialState.at(direction).totalPressure, testState.totalPressure,
                                     ", totalPressure");
  }
}
  // =========================================================================
  #endif  // MHD
#endif    // CUDA
