#include <vector>
#include <alps/bitops.h>
#include <limits>
#include <valarray>
#include <cassert>

/* Exact diagonalization of the Hamiltonian matrix using the Lanczos algorithm.
The size of the Hilbert space of an N-site system can be reduced using symmetries.
*/ 

class FermionBasis {
public:
typedef unsigned int state_type;
typedef unsigned int index_type;
