#pragma once

#include "../grid/grid3D.h"
#include "../global/global.h"
#include <stdio.h>


class FeedbackAnalysis {
    
    Real *h_circ_vel_x, *h_circ_vel_y;

    #ifdef PARTICLES_GPU
    Real *d_circ_vel_x, *d_circ_vel_y;
    void Compute_Gas_Velocity_Dispersion_GPU(Grid3D& G);
    #endif

    public: 
    int  countSN          {0};
    int  countResolved    {0};
    int  countUnresolved  {0};
    Real totalEnergy      {0};
    Real totalMomentum    {0};
    Real totalUnresEnergy {0};

    FeedbackAnalysis(Grid3D& G);
    ~FeedbackAnalysis();

    void Compute_Gas_Velocity_Dispersion(Grid3D& G);
    void Reset();
 
};