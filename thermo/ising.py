import numpy as np

"""
Ising model lattice encapsulation
"""

class IsingLattice:
    """
    Explore a simple nearest-neighbor 2D Ising model.
    This is an NxN periodic lattice in which each site
    has a particle with spin S (+/- 1). You can write the
    energy of the system as 
    E = -J Sum_{i,j} [S_{i,j}*(S_{I+1,j}+S_{i,j+1})]
        -H Sum_{i,j} [S_{i,j}]
    The periodic identification is the  site [i,j] = site [i+m*N,j+l*N] 
    for integers m,l. 
    """
    
    def __init__(self,N,J=1.0,H=0.0):
        self._N = N # lattice dimension
        self._J = J # interaction energy
        self._H = H # external magnetic field
        self.alignedspins() # set all spins to S (+/- 1)
        self._computeEM() # compute E and M over the lattice

    def alignedspins(self, S=1):
        """
        Set all spins to S (+/- 1)
        """
        if not(S==1 or S==-1):
            print("Error: spin must be +/-1")
            raise(AttributeError)
        self._spins = np.ones((self._N, self._N), dtype=int)*S
        self_computeEM()  

    def randomspins(self):
        """
        Randomize the spins.
        """
        self._spins = np.where(np.random.random((self._N, self._N)) > .5, 1, -1)
        self._computeEM()

    def N(self):
        """
        Return lattice dimension.
        """
        return self._N
    
    def M(self):
        """
        Return net magnetization.
        """
        return self._M
     
    def H(self):
        """
        Return exeternal magnetic field.
        """
        return self._H
    
    def J(self):
        """
        Return the interaction energy.
        """
        return self._J
    
    def E(self):
        """
        Return lattice energy.
        """
        return self._E

    def spins(self):
        """
        The following is a "trick" to help maintain the privacy of _spins.
        _spins is mutable, and the "return" without the ()*1 will simply
        return a pointer to _spins, and so could be changed external to the
        class. By multiplying by 1, a new array is created, and that is returned.
        """
        return (self._spins)*1
 
    def spinij(self, i, j):
        """
        Return spin of a specific point (i, j).
        """
        return self._spins[i%self._N, j%self._N]

    def __str__(self):
        """
        Query and return the lattice properties.
        """
        return "\nLattice properties: %d^2 cells, E=%f, M=%d, <E>=%f, <M>=%f\n"%\
               (self._N,self._E,self._M,self._E/self._N**2,self._M/self._N**2)

    def diagram(self):
        """
        Create a text diagram of the Ising model.
        """
        print(self)
        print(np.where(self._spins>0, "@"," "))

    def spinflip(self, i, j):
        """
        Flip the spins based on the interactions between them.
        """
        N = self._N # lattice dimension number
        s = self._spins # spins
        dM = -2.0*s[i%N, j%N]
        dE = 2.0*s[i%N, j%N]*self._H
        # For the case of N=1, the particle is its own neighbor so all the 
        # spins flip. The self-interaction E doesn't change. 
        if N>1:
            dE += 2*s[i%N, j%N]*self._J*(s[i%N, (j+1)%N]+s[(i+1)%N, j%N]+s[i%N, (j-1)%N]+s[(i-1)%N, j%N])
        self._spins[i%N, j%N] *= -1
        self._E += dE
        self._M += dM
        return dE

    def condspinflip(self, i, j, T): 
        """
        Spin flip at temperature T.
        """
        dE = self.spin_flip(i,j)
        if (dE<0.0 or (T>0.0 and (np.random.random()<np.exp(-dE/T)))): 
            return dE
        self.spinflip(i, j)
        return 0
    
    def _computeEM(self):
        """
        Compute the lattice energy E and net magnetization M of the lattice. 
        """ 
        self._M = np.sum(self._spins)*1.0
        self._E = 0.0
        for i in range(self._N):
            for j in range(self._N):
                self._E -= self._spins[i, j]*(self._spins[i, (j+1)%self._N]+self._spins[(i+1)%self._N, j])
        self._E *= self._J
        self._E -= self._M*self._H

    def mlln(x, data):
        """
        Compute the manual log likelihood normal, the likelihood the data given a new value,
        can compute that value. x is the tuple (mu, sigma) with mu the current value
        or sigma the new one. data is the observation of data so far.
        """
        return np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((data-x[0])**2) / (2*x[1]**2))

    def acceptance(x, xnew):
        """
        Do we accept the new value?
        """
        if xnew>x:
            return True
        else:
            accept=np.random.uniform(0,1)
            # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
            # less likely xnew are less likely to be accepted
            return (accept < (np.exp(xnew-x)))

    def methas(l, tm, pinit, data, it, arule):
        """
        For the likelihood model l, transition model tm, initial parameter pinit,
        data data, iterations it, and acceptance rule arule, return the accepted and rejected
        values from performing the Metropolis-Hastings (metropolis hastings) method
        on the data of the Ising model.
        """
        x = pinit 
        acc = [] # output accepted values
        rej = [] # rejected values
        for i in range(it):
            xnew = tm(x)
            xlik = l(x, data)
            xnewlik = l(xnew, data)
            if arule(xlik+np.log(prior(x)), xnewlik+np.log(prior(xnew))):
                x = xnew
                acc.append(xnew)
            else:
                rej.append(xnew)
        return [np.array(acc), np.array(rej)]
