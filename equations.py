from sympy import symbols, cos, sin, Derivative, simplify
from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols
from sympy.utilities import lambdify

# We are solving for a 2DoF arm which means two angles, two masses, two lengths,
# two moment of inertia, two torques etc...
m1, m2, l1, l2, i1, i2, g, t = symbols('m1 m2 l1 l2 i1 i2 g t')
theta1, theta2 = dynamicsymbols('theta1 theta2')

# (x1, y1) and (x2, y2) are the endpoints for each of the arm segments, we
# use basic math cos and sin functions to compute them - this is only used
# to make the center of mass formulas cleaner
x1 = l1 * cos(theta1)
y1 = l1 * sin(theta1)
# x2 = x1 + l2 * cos(theta1 + theta2)
# y2 = y1 + l2 * sin(theta1 + theta2)

# We need to know the center of mass of each link for the equations later,
# for simplicity we assume the center of mass is at the midpoint of each link
xcm1 = l1 / 2 * cos(theta1)
ycm1 = l1 / 2 * sin(theta1)
xcm2 = x1 + l2 / 2 * cos(theta1 + theta2)
ycm2 = y1 + l2 / 2 * sin(theta1 + theta2)

# We also needs the velocities for each center of mass, which is their
# differentiation in regards to t
vxcm1 = xcm1.diff(t)
vycm1 = ycm1.diff(t)
vxcm2 = xcm2.diff(t)
vycm2 = ycm2.diff(t)

# Similarly, we will need the theta dot and theta dot dot derivatives
theta1d = Derivative(theta1, t)
theta1dd = Derivative(theta1d, t)
theta2d = Derivative(theta2, t)
theta2dd = Derivative(theta2d, t)

# We now need to create the Lagrangian equations L = T - V, for this
# we need to sum the Kinetic and Potential energies of each link
# T = T_translational_1 + T_translational_2 + T_rotational_1 + T_rotational_2
# V = V_link_1 + V_link_2 where V = mgh
T1_trans = (1/2) * m1 * (vxcm1**2 + vycm1**2)
T2_trans = (1/2) * m2 * (vxcm2**2 + vycm2**2)
T1_rot = (1/2) * i1 * theta1d**2
T2_rot = (1/2) * i2 * (theta1d + theta2d)**2
T = T1_trans + T2_trans + T1_rot + T2_rot
V1 = m1 * g * ycm1
V2 = m2 * g * ycm2
V = V1 + V2
L = T - V

# Apparently, we also need to simplify the equations so as to help the solver
L = simplify(L)

# Now we can finally get the equations of motion
LM = LagrangesMethod(L, [theta1, theta2])
eqns = LM.form_lagranges_equations()

# As of now the equations cannot be used for math, we need to lambdify them
params = [m1, m2, l1, l2, i1, i2, g, theta1, theta1d, theta1dd, theta2, theta2d, theta2dd]
theta1_lambda = lambdify(params, eqns[0])
theta2_lambda = lambdify(params, eqns[1])
