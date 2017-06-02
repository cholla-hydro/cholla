/*! \file disk_ICs.cpp
 *  \brief Definitions of initial conditions for hydrostatic disks.
           Note that the grid is mapped to 1D as i + (x_dim)*j + (x_dim*y_dim)*k. */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <string.h>
#include <time.h>
#include "global.h"
#include "grid3D.h"
#include "mpi_routines.h"
#include "io.h"
#include "error_handling.h"



// function with logarithms used in NFW definitions
Real log_func(Real y)
{
  return log(1+y) - y/(1+y);
}

//vertical acceleration in NFW halo
Real gz_halo_D3D(Real R, Real z, Real *hdp)
{
  Real M_h = hdp[2]; //halo mass
  Real R_h = hdp[5]; //halo scale length
  Real c_vir = hdp[4]; //halo concentration parameter
  Real r = sqrt(R*R + z*z); //spherical radius
  Real x = r / R_h;
  Real z_comp = z/r;

  Real A = log_func(x);
  Real B = 1.0 / (r*r);
  Real C = GN*M_h/log_func(c_vir);

  //checked with wolfram alpha
  return -C*A*B*z_comp;
}


//radial acceleration in NFW halo
Real gr_halo_D3D(Real R, Real z, Real *hdp)
{
  Real M_h = hdp[2]; //halo mass
  Real R_h = hdp[5]; //halo scale length
  Real c_vir = hdp[4]; //halo concentration parameter
  Real r = sqrt(R*R + z*z); //spherical radius
  Real x = r / R_h;
  Real r_comp = R/r;

  Real A = log_func(x);
  Real B = 1.0 / (r*r);
  Real C = GN*M_h/log_func(c_vir);

  //checked with wolfram alpha
  return -C*A*B*r_comp;
}

//disk radial surface density profile
Real Sigma_disk_D3D(Real r, Real *hdp)
{
  //return the exponential surface density
  Real Sigma_0 = hdp[9];
  Real R_g     = hdp[10];
  Real R_c = 4;
  Real R_max = 5;
  Real Sigma;
  Sigma = Sigma_0 * exp(-r/R_g);
  // taper the edge of the disk to 0
  if (r > R_c) {
    Sigma *= exp(-(r - R_c)/(R_max-R_c));
  }
  return Sigma;
}

//vertical acceleration in miyamoto nagai
Real gz_disk_D3D(Real R, Real z, Real *hdp)
{
  Real M_d = hdp[1]; //disk mass
  Real R_d = hdp[6]; //MN disk length
  Real Z_d = hdp[7]; //MN disk height
  Real a = R_d;
  Real b = Z_d;
  Real A = sqrt(b*b + z*z);
  Real B = a + A;
  Real C = pow(B*B + R*R, 1.5);

  //checked with wolfram alpha
  return -GN*M_d*z*B/(A*C);
}

//radial acceleration in miyamoto nagai
Real gr_disk_D3D(Real R, Real z, Real *hdp)
{
  Real M_d = hdp[1]; //disk mass 
  Real R_d = hdp[6]; //MN disk length
  Real Z_d = hdp[7]; //MN disk height
  Real A = sqrt(Z_d*Z_d + z*z);
  Real B = R_d + A;
  Real C = pow(B*B + R*R, 1.5);

  //checked with wolfram alpha
  return -GN*M_d*R/C;
}


//NFW halo potential
Real phi_halo_D3D(Real R, Real z, Real *hdp)
{
  Real M_h = hdp[2]; //halo mass
  Real R_h = hdp[5]; //halo scale length
  Real c_vir = hdp[4]; //halo concentration parameter
  Real r = sqrt(R*R + z*z); //spherical radius
  Real x = r / R_h;

  Real C = GN*M_h/(R_h*log_func(c_vir));

  //limit x to non-zero value
  if(x<1.0e-9)
    x = 1.0e-9;

  //checked with wolfram alpha
  return -C*log(1+x)/x;
}

//Miyamoto-Nagai potential
Real phi_disk_D3D(Real R, Real z, Real *hdp)
{
  Real M_d = hdp[1]; //disk mass 
  Real R_d = hdp[6]; //MN disk length
  Real Z_d = hdp[7]; //MN disk height
  Real A = sqrt(z*z + Z_d*Z_d);
  Real B = R_d + A;
  Real C = sqrt(R*R + B*B);

  //patel et al. 2017, eqn 2
  return -GN*M_d/C;
}

//total potential
Real phi_total_D3D(Real R, Real z, Real *hdp)
{
  Real Phi_A = phi_halo_D3D(R,z,hdp);
  Real Phi_B = phi_disk_D3D(R,z,hdp);
  return Phi_A + Phi_B;
}

Real phi_hot_halo_D3D(Real r, Real *hdp)
{
  Real Phi_A = phi_halo_D3D(0,r,hdp);
  Real Phi_B = phi_disk_D3D(0,r,hdp);
  //return Phi_A;
  return Phi_A + Phi_B;
}


//returns the cell-centered vertical
//location of the cell with index k
//k is indexed at 0 at the lowest ghost cell
Real z_hc_D3D(int k, Real dz, int nz, int ng)
{
  //checked that this works, such that the
  //if dz = L_z/nz for the real domain, then the z positions
  //are set correctly for cell centers with nz spanning
  //the real domain, and nz + 2*ng spanning the real + ghost domains
  if(!(nz%2))
  {
    //even # of cells
    return 0.5*dz + ((Real) (k-ng-nz/2))*dz;
  }else{
    //odd # of cells
    return ((Real) (k-ng-(nz-1)/2))*dz;
  }
}

//returns the cell-centered radial
//location of the cell with index i
Real r_hc_D3D(int i, Real dr)
{
  //the zeroth cell is centered at 0.5*dr
  return 0.5*dr + ((Real) i)*dr;
}



/*! \fn void hydrostatic_ray_analytical_D3D(Real *rho, Real *r, Real *hdp, Real dr, int nr) 
 *  \brief Calculate the density at spherical radius r due to a hydrostatic halo. Uses an analytic 
    expression normalized by the value of the potential at the cooling radius. */
void hydrostatic_ray_analytical_D3D(Real *rho, Real *r, Real *hdp, Real dr, int nr)
{
  //Routine to determine the hydrostatic density profile
  //along a ray from the galaxy center
  int i;        //index along r direction

  Real gamma   = hdp[13]; //adiabatic index
  Real rho_eos = hdp[18]; //density where K_EOS is set
  Real cs      = hdp[19]; //sound speed at rho_eos
  Real r_cool  = hdp[20]; //cooling radius

  Real Phi_0; //potential at cooling radius

  Real D_rho; //ratio of density at mid plane and rho_eos

  Real gmo = gamma - 1.0; //gamma-1

  //compute the potential at the cooling radius
  Phi_0 = phi_hot_halo_D3D(r_cool,hdp);

  //We are normalizing to the central density
  //so D_rho == 1
  D_rho = 1.0;

  //store densities
  for(i=0;i<nr;i++)
  {
    r[i] = r_hc_D3D(i,dr);
    rho[i] = rho_eos*pow(D_rho - gmo*(phi_hot_halo_D3D(r[i],hdp)-Phi_0)/(cs*cs),1./gmo);
  }
}



/*! \fn void hydrostatic_column_analytical_D3D(Real *rho, Real R, Real *hdp, Real dz, int nz, int ng)
 *  \brief Calculate the 1D density distribution in a hydrostatic column. 
     Uses an iterative to scheme to determine the density at (R, z=0) relative to (R=0,z=0), 
     then sets the densities according to an analytic expression. */

void hydrostatic_column_analytical_D3D(Real *rho, Real R, Real *hdp, Real dz, int nz, int ng)
{
  //x is cell center in x direction
  //y is cell center in y direction
  //dz is cell width in z direction
  //nz is number of real cells
  //ng is number of ghost cells
  //total number of cells in column is nz * 2*ng
  //hdp[0] = M_vir; 
  //hdp[1] = M_d; 
  //hdp[2] = M_h; 
  //hdp[3] = R_vir; 
  //hdp[4] = c_vir; 
  //hdp[5] = R_s; 
  //hdp[6] = R_d; 
  //hdp[7] = z_d; 
  //hdp[8] = T_d; 
  //hdp[9] = Sigma_0; 
  //hdp[10] = R_g;
  //hdp[11] = H_g;
  //hdp[12] = K_eos;
  //hdp[13] = gamma;
  //hdp[14] = rho_floor;
  //hdp[15] = rho_eos;
  //hdp[16] = cs;

  int i,k;        //index along z axis
  Real z;     //cell center in z direction
  int nzt;      //total number of cells in z-direction
  Real Sigma; //surface density in column
  Real Sigma_n;
  Real z_min; //bottom of the cell
  Real z_max; //top of the cell
  Real Sigma_r; //surface density expected at r
  Real Sigma_0 = hdp[9]; //central surface density
  Real K = hdp[12]; //K coefficient in EOS; P = K \rho^gamma
  Real gamma = hdp[13];
  Real rho_floor = hdp[14]; //density floor

  Real rho_eos = hdp[15];
  Real cs      = hdp[16];

  Real Phi_0; //potential at z=0

  Real rho_0; //density at mid plane
  Real D_rho; //ratio of density at mid plane and rho_eos
  Real D_new; //new ratio of density at mid plane and rho_eos

  Real z_0, z_1, z_2; // heights for iteration
  Real z_disk_max;
  Real A_0, A_1; // density function to find roots

  //density integration
  Real phi_int, A;
  Real z_int_min, z_int_max, dz_int;
  Real Delta_phi;
  int n_int = 1000;

  //tolerance for secant method
  Real tol = 1.0e-6;

  //tolerance for surface density
  //fractional
  Real D_tol = 1.0e-5;


  int ks; //start of integrals above disk plane
  int km; //mirror of k
  if(nz%2)
  {
    ks = ng+(nz-1)/2;
  }else{
    ks = ng + nz/2;
  }

  // get the disk surface density
  // have verified that at this point, Sigma_r is correct
  Sigma_r = Sigma_disk_D3D(R, hdp);

  //set the z-column size, including ghost cells
  nzt = nz + 2*ng;

  //compute the mid plane potential
  Phi_0 = phi_total_D3D(R,0,hdp);

  //pick a fiducial guess for density ratio
  D_rho = pow( Sigma_r/Sigma_0, gamma-1. );

  //begin iterative determination of density field
  int flag = 0;
  int iter = 0; //number if iterations
  int flag_phi; //flag for density extent
  int iter_phi;

  D_new = D_rho;
  while(!flag)
  {
    //iterate the density ratio
    D_rho = D_new;

    //first determine the range of z where
    //the density above the central disk plane is
    //non-zero

    //get started, find the maximum extent of the disk
    iter_phi = 0;
    flag_phi = 0;
    z_0 = 1.0e-3;
    z_1 = 1.0e-2;
    while(!flag_phi)
    {
      A_0 = D_rho - (phi_total_D3D(R,z_0,hdp)-Phi_0)/(cs*cs);
      A_1 = D_rho - (phi_total_D3D(R,z_1,hdp)-Phi_0)/(cs*cs);
      z_2 = z_1 - A_1*(z_1-z_0)/(A_1-A_0);
      if( fabs(z_2-z_1)/fabs(z_1) > 10.)
        z_2 = 10.*z_1;
      //advance limit
      z_0 = z_1;
      z_1 = z_2;

      if(fabs(z_1-z_0)<tol)
      {
        flag_phi = 1;
        A_0 = D_rho - (phi_total_D3D(R,z_0,hdp)-Phi_0)/(cs*cs);
        A_1 = D_rho - (phi_total_D3D(R,z_1,hdp)-Phi_0)/(cs*cs);
        //make sure we haven't crossed 0
        if(A_1<0)
          z_1 = z_0;
      }
      iter_phi++;
      if(iter_phi>1000)
      {
        printf("Something wrong in determining central density...\n");
        printf("iter_phi = %d\n",iter_phi);
        printf("z_0 %e z_1 %e z_2 %e A_0 %e A_1 %e phi_0 %e phi_1 %e\n",z_0,z_1,z_2,A_0,A_1,phi_total_D3D(R,z_0,hdp),phi_total_D3D(R,z_1,hdp));
        MPI_Finalize();
        exit(0);
      }
    }
    A_1 = D_rho - (phi_total_D3D(R,z_1,hdp)-Phi_0)/(cs*cs);
    z_disk_max = z_1;

    //Compute surface density
    z_int_min = 0.0; //kpc
    z_int_max = z_1; //kpc
    dz_int = (z_int_max-z_int_min)/((Real) (n_int));
    phi_int = 0.0;
    for(k=0;k<n_int;k++)
    {
      z_0 = 0.5*dz_int + dz_int*((Real) k);
      Delta_phi = (phi_total_D3D(R,z_0,hdp)-Phi_0)/(cs*cs);
      A = D_rho - Delta_phi;
      phi_int += rho_eos*pow((gamma-1)*A,1./(gamma-1.))*dz_int;
    }

    //update density constant
    D_new = D_rho * pow(phi_int/(0.5*Sigma_r),1.-gamma);

    //if we have converged, exit!
    if(fabs(phi_int-0.5*Sigma_r)/(0.5*Sigma_r)<D_tol)
    {
      //done!
      flag = 1;
    }

    iter++;

    if(iter>100)
    {
      printf("About to exit...\n");

      MPI_Finalize();
      exit(0);
    }
  }

  //OK, at this stage we know how to set the densities
  //so let's take cell averages
  flag  = 0;
  n_int = 10; // integrate over a 1/10 cell
  for(k=ks;k<nzt;k++)
  {
    //find cell center, bottom, and top
    z_int_min  = z_hc_D3D(k,dz,nz,ng) - 0.5*dz;
    z_int_max  = z_hc_D3D(k,dz,nz,ng) + 0.5*dz;
    if(z_int_max>z_disk_max)
      z_int_max = z_disk_max;
    if(!flag)
    { 
      dz_int = (z_int_max-z_int_min)/((Real) (n_int));
      phi_int = 0.0;
      for(i=0;i<n_int;i++)
      {
        z_0 = 0.5*dz_int + dz_int*((Real) i) + z_int_min;
        Delta_phi = (phi_total_D3D(R,z_0,hdp)-Phi_0)/(cs*cs);
        A = D_rho - Delta_phi;
        phi_int += rho_eos*pow((gamma-1)*A,1./(gamma-1.))*dz_int;
      }

      //set density based on integral
      //of density in this cell
      rho[k] = phi_int/dz;

      if(z_int_max==z_disk_max)
      {
        flag = 1;
      }
    }else{
      //no mass up here!
      rho[k] = 0;
    }

    //mirror densities
    //above and below disk plane
    if(nz%2)
    {
      km = (ng+(nz-1)/2) - (k-ks);
    }else{
      km = ng + nz/2 - (k-ks) -1;
    }
    rho[km] = rho[k]; 
    Delta_phi = (phi_total_D3D(R,z_hc_D3D(k,dz,nz,ng),hdp)-Phi_0)/(cs*cs);
  }

  //check the surface density
  phi_int = 0.0;
  for(k=0;k<nzt;k++)
    phi_int += rho[k]*dz;
}


Real determine_rho_eos_D3D(Real cs, Real Sigma_0, Real *hdp)
{
  //OK, we need to set rho_eos based on the central surface density.
  //and the central potential
  int k;
  Real z_pos, rho_eos;
  Real Phi_0 = phi_total_D3D(0,0,hdp);
  Real gamma = hdp[13];
  Real Delta_phi;
  Real A = 0.0;

  //determine the maximum scale height by finding the
  //zero crossing of 1-(Phi-Phi_0)/cs^2
  //using the secant method
  Real z_0, z_1, z_2, A_0, A_1;
  Real tol = 1.0e-6;
  int flag_phi = 0;
  int iter_phi = 0;

  //get started
  z_0 = 1.0e-3;
  z_1 = 1.0e-2;
  while(!flag_phi)
  {
    A_0 = 1.0 - (phi_total_D3D(0,z_0,hdp)-Phi_0)/(cs*cs);
    A_1 = 1.0 - (phi_total_D3D(0,z_1,hdp)-Phi_0)/(cs*cs);
    z_2 = z_1 - A_1*(z_1-z_0)/(A_1-A_0);

    if( fabs(z_2-z_1)/fabs(z_1) > 10.)
      z_2 = 10.*z_1;

    //advance limit
    z_0 = z_1;
    z_1 = z_2;

    //printf("z_0 %e z_1 %e\n",z_0,z_1);
    if(fabs(z_1-z_0)<tol)
    {
      flag_phi = 1;
      A_0 = 1.0 - (phi_total_D3D(0,z_0,hdp)-Phi_0)/(cs*cs);
      A_1 = 1.0 - (phi_total_D3D(0,z_1,hdp)-Phi_0)/(cs*cs);
      //make sure we haven't crossed 0
      if(A_1<0)
        z_1 = z_0;
    }
    iter_phi++;
    if(iter_phi>1000)
    {
      printf("Something wrong in determining central density...\n");
      printf("iter_phi = %d\n",iter_phi);
      printf("z_0 %e z_1 %e z_2 %e A_0 %e A_1 %e phi_0 %e phi_1 %e\n",z_0,z_1,z_2,A_0,A_1,phi_total_D3D(0,z_0,hdp),phi_total_D3D(0,z_1,hdp));
      MPI_Finalize();
      exit(0);
    }
  }

  //generate a high resolution density and z profile
  int n_int = 1000;
  Real z_int_min = 0.0; //kpc
  Real z_int_max = z_1; //kpc
  Real dz_int = (z_int_max-z_int_min)/((Real) (n_int));
  Real phi_int = 0.0;

  //now integrate the density profile
  for(k=0;k<n_int;k++)
  {
    z_pos = 0.5*dz_int + dz_int*((Real) k);
    Delta_phi = phi_total_D3D(0,z_pos,hdp)-Phi_0;
    A = 1.0 - Delta_phi/(cs*cs);
    phi_int += pow((gamma-1)*A,1./(gamma-1.))*dz_int;
  }
  //use the potential integral to set central density at r=0
  rho_eos = 0.5*Sigma_0/phi_int;

  //check
  /*
  phi_int = 0.0;
  for(k=0;k<n_int;k++)
  {
    z_pos = 0.5*dz_int + dz_int*((Real) k);
    A = 1.0 - (phi_total_D3D(0,z_pos,hdp)-Phi_0)/(cs*cs);
    phi_int += rho_eos*pow((gamma-1)*A,1./(gamma-1.))*dz_int;
  }
  printf("phi_int %e Sigma_0/2 %e\n",phi_int,0.5*Sigma_0);
  */

  //return the central density
  return rho_eos;
}



Real halo_density_D3D(Real r, Real *r_halo, Real *rho_halo, Real dr, int nr)
{
  //interpolate the halo density profile
  int i;

  //find the index of the current
  //position in r_halo (based on r_hc_D3D)
  i = (int) ((r - 0.5*dr)/dr );
  if(i<0||i>=nr-1)
  {
    if(i<0)
    {
      i = 0;
    }else{
      i = nr-2;
    }
  }
  // return the interpolated density profile
  return (rho_halo[i+1] - rho_halo[i])*(r - r_halo[i])/(r_halo[i+1]-r_halo[i]) + rho_halo[i];
}





/*! \fn void Disk_3D(parameters P)
 *  \brief Initialize the grid with a 3D disk. */
void Grid3D::Disk_3D(parameters p)
{
  int i, j, k, id;
  Real x_pos, y_pos, z_pos, r, phi;
  Real d, n, a, a_d, a_h, v, vx, vy, vz, P, T_d, T_h, x;
  Real M_vir, M_h, M_d, c_vir, R_vir, R_s, R_d, z_d, Sigma;
  Real K_eos, rho_eos, cs, K_eos_h, rho_eos_h, cs_h;
  Real Sigma_0, R_g, H_g;
  Real rho_floor;
  Real r_cool;

  //M_vir = 1.0e12; // viral mass of MW in M_sun
  M_vir = 5.0e10; // viral mass of M82 in M_sun (guess)
  //M_d = 6.5e10; // mass of disk in M_sun (assume all stars)
  M_d = 1.0e10; // mass of M82 disk in M_sun (Greco 2012)
  M_h = M_vir - M_d; // halo mass in M_sun
  //R_d = 3.5; // MW stellar disk scale length in kpc
  R_d = 0.8; // M82 stellar disk scale length in kpc (Mayya 2009)
  //z_d = 3.5/5.0; // MW stellar disk scale height in kpc
  z_d = 0.15; // M82 stellar thin disk scale height in kpc (Lim 2013)
  //R_vir = 261; // MW viral radius in kpc
  R_vir = R_d/0.015; // M82 viral radius in kpc from R_(1/2) = 0.015 R_200 (Kravtsov 2013)
  //c_vir = 20; // MW halo concentration (to account for adiabatic contraction)
  c_vir = 10; // M82 halo concentration
  R_s = R_vir / c_vir; // halo scale length in kpc
  //T_d = 5.9406e5; // SET TO MATCH K_EOS SET BY HAND for K_eos   = 1.859984e-14 
  T_d = 2.0e5;
  T_h = 1.0e6; // halo temperature, at density floor 
  rho_eos = 1.0e7; //gas eos normalized at 1e7 Msun/kpc^3
  rho_eos_h = 3.0e3; //gas eos normalized at 3e3 Msun/kpc^3 (about n_h = 10^-3.5)

  R_g = 2.0*R_d;   //gas scale length in kpc
  //Sigma_0 = 0.25*M_d/(2*M_PI*R_g*R_g); //central surface density in Msun/kpc^2 (for MW)
  Sigma_0 = 0.25*M_d/(2*M_PI*R_g*R_g); //central surface density in Msun/kpc^2 (for M82)
  H_g = z_d; //initial guess for gas scale height
  rho_floor = 1.0e3; //ICs minimum density in Msun/kpc^3

  //cooling radius
  //r_cool = 157.0; //in kpc (MW)
  r_cool = 100.0; //in kpc (M82, guess)

  //EOS info
  cs = sqrt(KB*T_d/(0.6*MP))*TIME_UNIT/LENGTH_UNIT; //sound speed in kpc/kyr
  cs_h = sqrt(KB*T_h/(0.6*MP))*TIME_UNIT/LENGTH_UNIT; //sound speed in kpc/kyr

  //set some initial parameters
  int nhdp = 21;  //number of parameters to pass hydrostatic column
  Real *hdp = (Real *) calloc(nhdp,sizeof(Real));  //parameters
  hdp[0] = M_vir; 
  hdp[1] = M_d; 
  hdp[2] = M_h; 
  hdp[3] = R_vir; 
  hdp[4] = c_vir; 
  hdp[5] = R_s; 
  hdp[6] = R_d; 
  hdp[7] = z_d; 
  hdp[8] = T_d; 
  hdp[9] = Sigma_0; 
  hdp[10] = R_g;
  hdp[11] = H_g;
  hdp[13] = p.gamma;

  //determine rho_eos by setting central density of disk
  //based on central temperature
  rho_eos = determine_rho_eos_D3D(cs, Sigma_0, hdp);

  //set EOS parameters
  K_eos = cs*cs*pow(rho_eos,1.0-p.gamma)/p.gamma; //P = K\rho^gamma
  K_eos_h = cs_h*cs_h*pow(rho_eos_h,1.0-gama)/gama;

  //Store remaining parameters
  hdp[12] = K_eos;
  hdp[14] = 0.0; //rho_floor, set to 0
  hdp[15] = rho_eos;
  hdp[16] = cs;

  hdp[17] = K_eos_h;
  hdp[18] = rho_eos_h;
  hdp[19] = cs_h;
  hdp[20] = r_cool;


  //Now we can start the density calculation
  //we will loop over each column and compute
  //the density distribution
  int nz  = p.nz;
  int nzt = 2*H.n_ghost + nz;
  Real dz = p.zlen / ((Real) nz);
  Real *rho = (Real *) calloc(nzt,sizeof(Real));


  // create a look up table for the halo gas profile
  int nr = 1000;
  Real dr = sqrt(3)*0.5*fmax(p.xlen, p.zlen) / ((Real) nr);
  Real *rho_halo = (Real *) calloc(nr,sizeof(Real));
  Real *r_halo = (Real *) calloc(nr,sizeof(Real));
  Real rho_check = 0;


  //////////////////////////////////////////////
  //////////////////////////////////////////////
  // Produce a look up table for a hydrostatic hot halo
  //////////////////////////////////////////////
  //////////////////////////////////////////////
  hydrostatic_ray_analytical_D3D(rho_halo, r_halo, hdp, dr, nr);
  chprintf("Hot halo lookup table generated...\n");


  //////////////////////////////////////////////
  //////////////////////////////////////////////
  // Add a disk component
  //////////////////////////////////////////////
  //////////////////////////////////////////////

  // compute a
  // hydrostatic column for the disk 
  // and add the disk density and thermal energy
  // to the density and energy arrays
  for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
    chprintf("j %d\n",j);
    for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

      // get the centered x, y, and z positions
      k = H.n_ghost + H.ny;
      Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);

      //cylindrical radius
      r = sqrt(x_pos*x_pos + y_pos*y_pos);


      //Compute the hydrostatic density profile in this z column
      //owing to the disk
      hydrostatic_column_analytical_D3D(rho, r, hdp, dz, nz, H.n_ghost);

      //store densities
      for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
        id = i + j*H.nx + k*H.nx*H.ny;

        //get density from hydrostatic column computation
        d = rho[nz_local_start + H.n_ghost + (k-H.n_ghost)];

        // set pressure adiabatically
        P = K_eos*pow(d,p.gamma);

        // store density in density
        C.density[id]    = d;

        // store internal energy in Energy array
        C.Energy[id] = P/(gama-1.0);
      }
    }
  }

  int idm, idp;
  Real xpm, xpp;
  Real ypm, ypp;
  Real zpm, zpp;
  Real Pm, Pp;

  //pressure gradients for changing
  //the rotational velocity
  Real dPdx, dPdy, dPdr;

  //compute radial pressure gradients, adjust circular velocities
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        id = i + j*H.nx + k*H.nx*H.ny;

        //get density
        d = C.density[id];

        //restrict to regions where the density
        //has been set
        if(d>0.0)
        {
          // get the centered x, y, and z positions
          Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
          // calculate radial position and phi (assumes disk is centered at 0, 0)
          r = sqrt(x_pos*x_pos + y_pos*y_pos);
          phi = atan2(y_pos, x_pos); // azimuthal angle (in x-y plane)

          // radial acceleration from disk
          a_d = fabs(gr_disk_D3D(r, z_pos, hdp));
          // radial acceleration from halo 
          a_h = fabs(gr_halo_D3D(r, z_pos, hdp));

          //  pressure gradient along x direction
          // gradient calc is first order at boundaries
          if (i == H.n_ghost) idm = i + j*H.nx + k*H.nx*H.ny;
          else idm  = (i-1) + j*H.nx + k*H.nx*H.ny; 
          if (i == H.nx-H.n_ghost-1) idp  = i + j*H.nx + k*H.nx*H.ny;
          else idp  = (i+1) + j*H.nx + k*H.nx*H.ny; 
          Get_Position(i-1, j, k, &xpm, &ypm, &zpm);
          Get_Position(i+1, j, k, &xpp, &ypp, &zpm);
          Pm = C.Energy[idm]*(gama-1.0); // only internal energy stored in energy currently
          Pp = C.Energy[idp]*(gama-1.0); // only internal energy stored in energy currently
          dPdx =  (Pp-Pm)/(xpp-xpm);

          //pressure gradient along y direction
          if (j == H.n_ghost) idm = i + j*H.nx + k*H.nx*H.ny;
          else idm  = i + (j-1)*H.nx + k*H.nx*H.ny; 
          if (j == H.ny-H.n_ghost-1) idp  = i + j*H.nx + k*H.nx*H.ny; 
          else idp  = i + (j+1)*H.nx + k*H.nx*H.ny; 
          Get_Position(i, j-1, k, &xpm, &ypm, &zpm);
          Get_Position(i, j+1, k, &xpp, &ypp, &zpm);
          Pm = C.Energy[idm]*(gama-1.0); // only internal energy stored in energy currently
          Pp = C.Energy[idp]*(gama-1.0); // only internal energy stored in energy currently
          dPdy =  (Pp-Pm)/(ypp-ypm);

          //radial pressure gradient
          dPdr = x_pos*dPdx/r + y_pos*dPdy/r;

          //radial acceleration
          a = a_d + a_h + dPdr/d;
        
          if(isnan(a)||(a!=a)||(r*a<0))
          {
            printf("i %d j %d k %d a %e a_d %e dPdr %e d %e\n",i,j,k,a,a_d,dPdr,d);
            printf("i %d j %d k %d x_pos %e y_pos %e z_pos %e dPdx %e dPdy %e\n",i,j,k,x_pos,y_pos,z_pos,dPdx,dPdy);
            printf("i %d j %d k %d Pm %e Pp %e\n",i,j,k,Pm,Pp);
            printf("ypp %e ypm %e xpp %e zpm %e r %e\n",ypp,ypm, xpp, xpm ,r);
            printf("Energy pm %e pp %e density pm %e pp %e\n",C.Energy[idm],C.Energy[idp],C.density[idm],C.density[idp]);
          }
          else {
      
            // radial velocity 
            v = sqrt(r*a);
            vx = -sin(phi)*v;
            vy = cos(phi)*v;
            vz = 0;

            // set the momenta 
            C.momentum_x[id] = d*vx;
            C.momentum_y[id] = d*vy;
            C.momentum_z[id] = d*vz;

            //sheepishly check for NaN's!

            if((d<0)||(P<0)||(isnan(d))||(isnan(P))||(d!=d)||(P!=P))
              printf("d %e P %e i %d j %d k %d id %d\n",d,P,i,j,k,id);

            if((isnan(vx))||(isnan(vy))||(isnan(vz))||(vx!=vx)||(vy!=vy)||(vz!=vz)) {
              printf("vx %e vy %e vz %e i %d j %d k %d id %d\n",vx,vy,vz,i,j,k,id);
            }
            else {
            //if the density is negative, there
            //is a bigger problem!
              if(d<0)
              {
                printf("pid %d error negative density i %d j %d k %d d %e\n",-1,i,j,k,d);
              }
            }
          }
        } 
      }
    }
  }


  //////////////////////////////////////////////
  //////////////////////////////////////////////
  // Add a hot, hydrostatic halo
  //////////////////////////////////////////////
  //////////////////////////////////////////////
  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        // get the cell index
        id = i + j*H.nx + k*H.nx*H.ny;

        // get the centered x, y, and z positions
        Get_Position(i, j, k, &x_pos, &y_pos, &z_pos);
        
        // calculate 3D radial position and phi (assumes halo is centered at 0, 0)
        r = sqrt(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos);

        // interpolate the density at this position
        d = halo_density_D3D(r, r_halo, rho_halo, dr, nr);

        // set pressure adiabatically
        P = K_eos_h*pow(d,p.gamma);

        // store density in density
        C.density[id] += d;

        // store internal energy in Energy array
        C.Energy[id] += P/(gama-1.0);

      }
    } 
  }

  //////////////////////////////////////////////
  //////////////////////////////////////////////
  // Add kinetic energy to total energy
  //////////////////////////////////////////////
  //////////////////////////////////////////////

  for (k=H.n_ghost; k<H.nz-H.n_ghost; k++) {
    for (j=H.n_ghost; j<H.ny-H.n_ghost; j++) {
      for (i=H.n_ghost; i<H.nx-H.n_ghost; i++) {

        id = i + j*H.nx + k*H.nx*H.ny;
        
        // add kinetic contribution to total energy
        C.Energy[id] += 0.5*(C.momentum_x[id]*C.momentum_x[id] + C.momentum_y[id]*C.momentum_y[id] + C.momentum_z[id]*C.momentum_z[id])/C.density[id];

      }
    } 
  }

  //free density profile
  free(rho);
  free(hdp);

  //free the arrays 
  //for the hot halo
  //gas lookup table
  free(r_halo);
  free(rho_halo);

}


