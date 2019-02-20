from __future__ import division
from sympy import *
from sympy.physics.mechanics import *

"""
Initialize KanesMethod object and use kanes_equation class method linearization.
"""

q1 = dynamicsymbols("q1") # Pendulum angle
u1 = dynamicsymbols("u1") # Anguluar velocity
q1d = dynamicsymbols("q1", 1)
L, m, t, g = symbols("L, m, t, g")

# World frame
N = ReferenceFrame("N")
pN = Point("N*")
pN.set_vel(N, 0)

# A.x is along the pendulum
A = N.orientview("A", "axis", [q1, N.z])
A.set_ang_vel(N, u1*N.z)

# Locate point P relative to origin N*
P = pN.locatenew("P", L*A.x)
vel_P = P.v2pt_theory(pN, N, A)
pP = Particle("pP", P, m)

# Create Kinematic Differential Equations
kde = Matrix([q1d - u1])

# Input force at P
R = m*g*N.x

# Solve for eom with KanesMethod
KM = KanesMethod(N, q_ind=[q1], u_ind=[u1], kd_eqs=kde)
fr, frstar = KM.kanes_equations([pP], [P, R])


"""
Linear Lagrange's Equations
"""
A = N.orientnew("A", "axis", [q1, N.z])
A.set_ang_vel(N, q1d*N.z)
