#ifndef PHASE_FIELD_CRYSTAL_SYSTEM
#define PHASE_FIELD_CRYSTAL_SYSTEM

#include <deal.II/grid/tria.h>

template <int dim>
class PhaseFieldCrystalSystem
{
public:
    PhaseFieldCrystalSystem();
    void run(unsigned int n_refines);

private:
    dealii::Triangulation<dim> triangulation;
};

#endif
