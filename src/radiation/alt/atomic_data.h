/*LICENSE*/

#ifndef PHYSICS_ATOMIC_DATA_H
#define PHYSICS_ATOMIC_DATA_H

//
//  Various atomic, chemical, and thermal rates.
//
#include "atomic_data_decl.h"


namespace Physics
{
    namespace AtomicData
    {
        //
        //  Cross-sections are in barns (to limit the numeric range and fit into float).
        //  The x-axis in the table (frequency) is represented as xi = log(hnu/1Ry).
        //
        const CrossSection* CrossSections();
        const CrossSection* CrossSectionsGPU();

        void Create();
        void Delete();
    };
};

#endif // PHYSICS_ATOMIC_DATA_H
