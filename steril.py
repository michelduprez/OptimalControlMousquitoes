from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
m = GEKKO()

plt.rcParams['font.size'] = 15

# Parameters
nu_float = 0.5
beta_E_float=10
gamma_float=1.0/3.0
tau_E_float=0.05
delta_E_float=0.03
beta_F_float=0.01
delta_F_float=0.04
delta_s_float=0.12
F_star = 5106.0
E_star = F_star*delta_F_float/(nu_float*beta_F_float)
K_float = E_star/(1.0-(tau_E_float+delta_E_float)*delta_F_float/(nu_float*beta_E_float*beta_F_float))
nu=m.Param(value=nu_float)
beta_E=m.Param(value=beta_E_float)
gamma=m.Param(value=gamma_float)
tau_E=m.Param(value=tau_E_float)
delta_E=m.Param(value=delta_E_float)
beta_F=m.Param(value=beta_F_float)
delta_F=m.Param(value=delta_F_float)
delta_s=m.Param(value=delta_s_float)
K=m.Param(value=K_float)
T = 80
C = 150000
U_bar = 20000


# Time interval
nt = 700
m.time = np.linspace(0,T,nt)
#delta_t = T/nt


# Control
u = m.MV(lb=0,ub=U_bar)
u.STATUS = 1
u.DCOST = 0


# Variables
E = m.Var(value=E_star)
F = m.Var(value=F_star)
M_s = m.Var(value=0)
U = m.Var(value=0)


# Vector to extract the final value
final_array = np.zeros(nt)
final_array[-1] = 1.0
final = m.Param(value=final_array)


# Equations
m.Equation(E.dt()==beta_E*F*(1-E/K)*F/(F+gamma*M_s)-(tau_E+delta_E)*E)
m.Equation(F.dt()==nu*beta_F*E-delta_F*F)
m.Equation(M_s.dt()==u-delta_s*M_s)
m.Equation(U.dt()==u)
m.Equation(final*U<=C)


# Objective Function
m.Obj(0.5*((final*E)**2+(final*F)**2))


# Resolution
m.options.IMODE = 6
m.options.NODES = 4
m.options.MV_TYPE = 1
m.options.SOLVER = 3
print('begining of computations')
m.solve()
print('end of computations')


# Print results
E_array=np.transpose(np.matrix(E))
F_array=np.transpose(np.matrix(F))
M_s_array=np.transpose(np.matrix(M_s))
u_array=np.transpose(np.matrix(u))
U_array=np.transpose(np.matrix(U))
t_array = np.arange(nt)*T/nt
J = float(0.5*(E_array[-1]**2+F_array[-1]**2))
ET = float(E_array[-1])
FT = float(F_array[-1])
UT = float(U_array[-1])
E0 = float(E_array[0])
F0 = float(F_array[0])
print('Objective = int u: ' + str(J))
print('E(T): ' + str(ET))
print('F(T): ' + str(FT))
print('int u: ' + str(UT))
print('E(0): ' + str(E0))
print('F(0): ' + str(F0))
print('K: ' + str(K_float))


# Plot
plt.figure(1)
plt.plot(t_array,E_array,'g-',linewidth=2,label=r'$E$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.axis([0, T, min(E_array)-0.05*(max(E_array)-min(E_array)), max(E_array)+0.05*(max(E_array)-min(E_array))])
plt.savefig("simu_steril_E_{name0}_{name1}_{name2}.png".format(name0 = str(C),name1 = str(U_bar),name2 = str(gamma_float)))
plt.figure(2)
plt.plot(t_array,F_array,'r-',linewidth=2,label=r'$F$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.axis([0, T, min(F_array)-0.05*(max(F_array)-min(F_array)), max(F_array)+0.05*(max(F_array)-min(F_array))])
plt.savefig("simu_steril_F_{name0}_{name1}_{name2}.png".format(name0 = str(C),name1 = str(U_bar),name2 = str(gamma_float)))
plt.figure(3)
plt.plot(t_array,u_array,'b-',linewidth=2,label=r'$u$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.axis([0, T, 0, max(u_array)+0.05*(max(u_array)-min(u_array))])
plt.savefig("simu_steril_u_{name0}_{name1}_{name2}.png".format(name0 = str(C),name1 = str(U_bar),name2 = str(gamma_float)))
plt.show()
