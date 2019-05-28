#include <vector>
#include <alps/bitops.h>
#include <limits>
#include <valarray>
#include <cassert>

/* Exact diagonalization of the Hamiltonian matrix using the Lanczos algorithm.*/ 

class FermionBasis {
public:
typedef unsigned int state_type;
typedef unsigned int index_type;
