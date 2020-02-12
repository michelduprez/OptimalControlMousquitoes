from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
m = GEKKO()

plt.rcParams['font.size'] = 15

# Parameters
nu_float = 0.5
beta_E_float = 10.0
tau_E_float=0.05
delta_E_float=0.03
beta_F_float=0.01
s_h_float = 0.9951
delta_F_float = 0.04
eta_float = 0.95
delta_float = 1.25
F_star = 5106.0
b_float = nu_float*beta_F_float*beta_E_float/(tau_E_float+delta_E_float)#3.125
print('b='+str(b_float))
K_float = F_star/(nu_float*beta_F_float/delta_F_float-nu_float*beta_F_float/b_float)
print('K='+str(K_float))
nu=m.Param(value=nu_float)
beta_E=m.Param(value=beta_E_float) 
tau_E=m.Param(value=tau_E_float)
delta_E=m.Param(value=delta_E_float)
beta_F=m.Param(value=beta_F_float)
b=m.Param(value=b_float) 
s_h=m.Param(value=s_h_float)
delta_F=m.Param(value=delta_F_float)
eta=m.Param(value=eta_float)
delta=m.Param(value=delta_float)
K=m.Param(value=K_float)
T = 80
C = 1000
U_bar = 150


# Time interval
nt = 500
m.time = np.linspace(0,T,nt)
delta_t = T/nt


# Control
u = m.MV(lb=0,ub=U_bar)
u.STATUS = 1
u.DCOST = 0


# Variables
E_u = m.Var(value=K*(1-delta_F/b))
F_u = m.Var(value=F_star)
E_i = m.Var(value=0.0)
F_i = m.Var(value=0.0)
U = m.Var(value=0.0)


# Vector to extract the final value
final_array = np.zeros(nt)
final_array[-1] = 1.0
final = m.Param(value=final_array)


# Equations
m.Equation(E_u.dt()==beta_E*F_u*(1-s_h*F_i/(F_u+F_i))*(1-(E_u+E_i)/K)-(tau_E+delta_E)*E_u)
m.Equation(F_u.dt()==nu*beta_F*E_u-delta_F*F_u)
m.Equation(E_i.dt()==eta*beta_E*F_i*(1-(E_u+E_i)/K)-(tau_E+delta_E)*E_i)
m.Equation(F_i.dt()==nu*beta_F*E_i-delta*delta_F*F_i+u)
m.Equation(U.dt()==u)
m.Equation(final*U<=C)


# Objective Function
m.Obj(0.5*(final*E_u)**2+0.5*(0.5*(abs(K*(1-delta*delta_F/(b*eta))-final*E_i)+K*(1-delta*delta_F/(b*eta))-final*E_i))**2+0.5*(final*F_u)**2+0.5*(0.5*(abs(K*(nu*beta_F/(delta*delta_F)-nu*beta_F/(b*eta))-final*F_i)+K*(nu*beta_F/(delta*delta_F)-nu*beta_F/(b*eta))-final*F_i))**2)


# Resolution
m.options.IMODE = 6
m.options.NODES = 4
m.options.MV_TYPE = 1
m.options.SOLVER = 3
print('begining of computations')
m.solve()
print('end of computations')


# Print results
E_u_array=np.transpose(np.matrix(E_u))
F_u_array=np.transpose(np.matrix(F_u))
E_i_array=np.transpose(np.matrix(E_i))
F_i_array=np.transpose(np.matrix(F_i))
u_array=np.transpose(np.matrix(u))
U_array=np.transpose(np.matrix(U))
t_array = np.arange(nt)*T/nt
E_u_T = float(final_array*E_u_array)
F_u_T = float(final_array*F_u_array)
E_i_T = float(final_array*E_i_array)
F_i_T = float(final_array*F_i_array)
print(r'$E_u(T):$ ' + str(E_u_T))
print(r'$F_u(T):$ ' + str(F_u_T))
print(r'$E_i(T):$ ' + str(E_i_T))
print(r'$F_i(T):$ ' + str(F_i_T))


# Plot
plt.figure(1)
plt.plot(t_array,F_u_array,'g-',linewidth=2,label=r'$F_u$')
plt.plot(t_array,F_i_array,'r-',linewidth=2,label=r'$F_i$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.savefig("simu_wolbachia_F_{name0}_{name1}.png".format(name0 = str(C),name1 = str(U_bar)))
plt.figure(2)
plt.plot(t_array,E_u_array,'g-',linewidth=2,label=r'$E_u$')
plt.plot(t_array,E_i_array,'r-',linewidth=2,label=r'$E_i$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.savefig("simu_wolbachia_E_{name0}_{name1}.png".format(name0 = str(C),name1 = str(U_bar)))
plt.figure(3)
plt.plot(t_array,u_array,'b-',linewidth=2,label=r'$u$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.axis([0, T,0, max(u_array)+0.05*(max(u_array)-min(u_array))])
plt.savefig("simu_wolbachia_u_{name0}_{name1}.png".format(name0 = str(C),name1 = str(U_bar)))
plt.show()
