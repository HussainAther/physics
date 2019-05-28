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
    FermionBasis (int L, int N);
    state_type state(index_type i) const {return states_[i];}
    index_type index(state_type s) const {return index_[s];}
    unsigned int dimension() const { return states_.size();}
private:
    std::vector<state_type> states_;
    std::vector<index_type> index_;
};

/* Fermion basis states */
FermionBasis::FermionBasis(int L, int N)
{
    index_.resize(1<<L); // 2^L entries
    for (state_type s=0;s<index_.size();++s)
        if(alps::popcnt(s)==N) {
            // correct number of particles
            states_.push_back(s);
            index_[s]=states_.size()-1;
}
    else
        // invalid state
        index_[s]=std::numeric_limits<index_type>::max();
}
