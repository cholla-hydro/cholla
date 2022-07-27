#pragma once

#include "../grid/grid3D.h"
#include "../global/global.h"
#include <stdio.h>


class FeedbackAnalysis {
    
    Real *h_circ_vel_x, *h_circ_vel_y;
    #ifdef PARTICLES_GPU
    Real *d_circ_vel_x, *d_circ_vel_y;
    #endif

    #ifdef PARTICLES_GPU
    void Compute_Gas_Velocity_Dispersion_GPU(Grid3D& G);
    #endif

    public: 
    int countSN;
    int countResolved;
    int countUnresolved;
    Real totalEnergy;
    Real totalMomentum;
    Real totalUnresEnergy;

    FeedbackAnalysis(Grid3D& G);
    ~FeedbackAnalysis();

    void Compute_Gas_Velocity_Dispersion(Grid3D& G);
    void Reset();
 
};