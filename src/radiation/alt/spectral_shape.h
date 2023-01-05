/*LICENSE*/

#ifndef PHYSICS_RT_SPECTRAL_SHAPE_H
#define PHYSICS_RT_SPECTRAL_SHAPE_H

#ifndef COMMON_INCLUDED
#error "Missing common.h include."
#endif


//
//  Various atomic, chemical, and thermal rates.
//
namespace Physics
{
    namespace RT
    {
        //
        //  Spectral shapes use the same frequency axis as cross sections.
        //  Shapes are proportional to \dot{N}_\xi, normalized such that \int_Ry g_\xi d\xi = 1
        //  Shapes are not pre-computed to allow parameters.
        //
        class SpectralShape
        {

        public:

            //
            //  PopII and PopIII shapes from Ricotti et al 2002ApJ...575...33R.
            //  Used in CROC.
            //
            static void PopII(STL::Vector<float>& s);
            static void PopIII(STL::Vector<float>& s);

            //
            //  Average QSO spectral shape from Richards et al 2006ApJS..166..470R and Hopkins et al 2007, ApJ, 654, 731  (i.e. NG's fit to it).
            //  Used in CROC.
            //
            static void Quasar(STL::Vector<float>& s);

            //
            //  Black body spectrum.
            //
            static void BlackBody(float tem, STL::Vector<float>& s);

        private:

            static void Normalize(STL::Vector<float>& s);
        };
    };
};

#endif // PHYSICS_RT_SPECTRAL_SHAPE_H
