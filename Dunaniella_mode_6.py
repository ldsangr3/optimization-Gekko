
#%% Bennatia Model Dynamic Optimizacion 
# Viyils Sangregorio Sot  v 1.1
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
# latex

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# Configuration axis
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

#Initialize model
m = GEKKO (remote=True)

# Folder path 
m._path = r'C:\Users\Viyils\OneDrive\Optimization Python\Benanttia_Optimization_time\results'

#%% Parameters 

mu_   = m.Param(value=2, name="u");
rho_m = m.Param(value=9.3, name="rhom");
K_Q   = m.Param(value=1.8, name="KQ");
K_s   = m.Param(value=0.105, name="KS");
S_in  = m.Param(value=100, name="S_in");
K_sI  = m.Param(value=150, name="KsI");
K_iI  = m.Param(value=2000, name="KiI");
mu_I  = m.Param(value=0.6461, name="muI");

# Time 
nt =500
m.time = np.linspace(0,150,nt)

# Define Dilution rate as Manipulated variable
D = m.MV(value=0.1, lb=0.001, ub=2);
# Manipulated variable
D.STATUS = 1 # allow optimizer to change
D.DCOST = 0.1 # smooth out D movement
D.DMAX = 0.1   # slow down D
#%% Variables
# Constrains Equations
# x > 0, Q < 8.9969 , Q > 1.8, s > 0, s < 120 
x = m.Var(value=0.2, lb=0 , ub=100, name ='x');
Q = m.Var(value=1.8, lb=1.8 , ub=8.9969, name = 'Q');
s = m.Var(value=0.01, lb=0 , ub=100, name = 's') ;
mu = m.Var(value=0.1, lb=0 , ub=2, name = 'mut');
#time
#t=m.Var(value=0) #for time optimization

tf = 40 
nt =15*tf+1
m.time = np.linspace(0,tf,nt)
t = m.Param(value=m.time); # For maximun value

#%% Equations 
# m.Equation(t.dt()  == 1) # For time optimization only
#For implicit form x.dt() Q.dt() s.dt()
m.Equation(mu == mu_*(1 - K_Q/(Q) )*mu_I);
m.Equation(x.dt()  == mu*x - D*x);
m.Equation(Q.dt()  == rho_m*((s)/ ((s)+ K_s)) - mu*Q);
m.Equation(s.dt()  == (S_in - s)*D - rho_m*((s)/ ((s)+ K_s)) * x);

#%% Constrains Equations
#m.Equations([ x > 0, Q < 7.19 , Q > 1.8, s > 0, s < 120 ]);

m.Maximize(D*x)
#%% Solver options

m.options.SENSITIVITY = 1   # sensitivity analysis

m.options.IMODE = 6
m.solve(disp=True,GUI=True)

# 1. Steady-state simulation (SS)
# 2. Model parameter update (MPU)
# 3. Real-time optimization (RTO)
# 4. Dynamic simulation (SIM)
# 5. Moving horizon estimation (EST)
# 6. Nonlinear control / dynamic optimization (CTL)
# 7. Sequential dynamic simulation (SQS)
# 8. Sequential dynamic estimation (SQE)
# 9. Sequential dynamic optimization (SQO)
#%% Solve
m.solve();


#configuration
# Plot solution
plt.figure()
plt.plot(m.time,x.value,"r-",LineWidth=2)
plt.ylabel(r" Biomass " r"($ \mu m^3 \cdot L^{-1}$)", fontsize=25)
plt.xlabel(r" Time " r"($ days$)", fontsize=25)

# Maximal point
# Final time before something weird at the end 
Final = 5
ymax = max(x.value)
xmax = 15
text= r"$Optimal \ x^*(t) = {:.1f}$".format(ymax) + r" ($ \mu m^3 \cdot L^{-1} $) "
plt.annotate(text , xy=(xmax, ymax+5),
fontsize=22, 
arrowprops=dict(facecolor='black', shrink=0.05),)
plt.xlim(0, tf - Final) 
plt.grid(True)
plt.show()
print(m.path)
print('Dilution rate is: ' + str(D.value[nt]))

#thanks to Sandra Milena Rodriguez and Edgar Mayorga el profe
