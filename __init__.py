import numpy as np
import matplotlib.pylab as plt
from sympy import symbols, init_printing, sqrt, simplify, Matrix, pprint, exp, log

class TwoQVoterModel(object):
    
    def __init__(self,**kwargs):
        """
        Define a 2qVM with parameters N, zp, zm, s1, s2, q1, q2, rho.
        """
        for key, value in kwargs.items():
            setattr(self,key,value)
        assert self.N==self.zp+self.zm+self.s1+self.s2, "Number of agents must add up to population size"
        assert isinstance(self.q1, int) & isinstance(self.q2, int), "q1 and q2 must be integers"
        assert (self.rho>=0) & (self.rho<=1), "Density of +voters needs to be in (0,1)"
        
        self.setup()

    def setup(self):
        """ Initialises all the required variables to track the simulation """
        
        self.s1, self.s2 = self.s1/self.N, self.s2/self.N
        self.zp, self.zm = self.zp/self.N, self.zm/self.N
            
        self.s = 1 - self.zp - self.zm 
        self.x1, self.x2 = self.rho*self.s1, self.rho*self.s2 
        self.dx = 1.0/self.N
        self.x1_track = [self.x1]
        self.x2_track = [self.x2]
        self.set_transition_matrix()

    def set_transition_matrix(self):
        """ Calculates the transition rates """
        x1 = self.x1_track[-1]
        x2 = self.x2_track[-1]
        s1,s2,q1,q2 = self.s1, self.s2, self.q1, self.q2
        zp,zm = self.zp, self.zm
        self.Tr = [(s1-x1)*(zp+x1+x2)**q1, # x1 -> x1 + 1
                   (x1)*(zm+s1+s2-(x1+x2))**q1, # x1 -> x1 - 1
                   (s2-x2)*(zp+x1+x2)**q2, # x2 -> x2 + 1
                   (x2)*(zm+s1+s2-(x1+x2))**q2] # x2 -> x2 - 1
        
    def fixed_points(self):
        """Returns the fixed points of the system [(x1,x2,stability)]. Valid only for s1=s2:=s, zp=zm:=z, """
        fps = []
        s=self.s1
        z=self.zp
        fp1 = s/2
        fps.append((fp1,fp1))
        if z < 1/6:
            fp2a,fp2b = (s/(2*(1-s)))*(1-s-(s*(4-3*s)-1)**0.5), 0.5*(s-(s*(4-3*s)-1)**0.5)
            fp3a,fp3b = (s/(2*(1-s)))*(1-s+(s*(4-3*s)-1)**0.5), 0.5*(s+(s*(4-3*s)-1)**0.5) 
            fps.append((fp2a,fp2b))
            fps.append((fp3a,fp3b))
        return fps
        
        
    def run_iteration(self):
        """ Run one iteration of the model """
        r1 = np.random.random()
        cumsum = np.cumsum(self.Tr)
        ix = np.searchsorted(cumsum,r1)
        
        if ix == 0:
            self.x1_track.append(self.x1_track[-1] + self.dx)
            self.x2_track.append(self.x2_track[-1])
        elif ix == 1:
            self.x1_track.append(self.x1_track[-1] - self.dx)
            self.x2_track.append(self.x2_track[-1])
        elif ix == 2:
            self.x1_track.append(self.x1_track[-1])
            self.x2_track.append(self.x2_track[-1] + self.dx)
        elif ix == 3:
            self.x1_track.append(self.x1_track[-1])
            self.x2_track.append(self.x2_track[-1] - self.dx)
        else:
            self.x1_track.append(self.x1_track[-1])
            self.x2_track.append(self.x2_track[-1])
        
        self.set_transition_matrix()
        
    def angular_velocity(self, t, T, burn=10000):
        angvec = [self.x1_track[tau]*self.x2_track[tau+t] - self.x1_track[tau+t]*self.x2_track[tau] for tau in range(burn,burn+T-t)] 
        angvec = sum(angvec)/(T-t)
        return angvec
    
    def calculate_angular_momentum(self,t_range):
            
        s, z, N, tau = symbols('s z N tau')
        q1, q2 = symbols('q_1 q_2')
        parameters = dict(s=self.s1, q_1=self.q1, q_2=self.q2, z=self.zp, N=self.N)

        if self.zp < 1/6:
            # Lower FP
            x1 = (s/(2*s*(1-s)))*(1-s-sqrt(s*(4-3*s)-1))
            x2 = (s-sqrt(s*(4-3*s)-1))/2
            x1 = simplify(x1.subs(dict(q_1=1,q_2=2)))
            x2 = simplify(x2.subs(dict(q_1=1,q_2=2)))

        else:    
            # Middle FP
            x1 = s/2
            x2 = s/2

        mu = simplify(z + x1 + x2)
        X = { 1:simplify(q1*(s - x1)*(mu**q1)/(mu*(1-mu))),   # X_1mu 
              2:simplify(q2*(s - x2)*(mu**q2)/(mu*(1-mu))) }  # X_2mu

        F = Matrix([[ mu**q1+(1-mu)**q1-X[1], -X[1]],
                  [ -X[2], mu**q2+(1-mu)**q2-X[2]]])

        w1 = (s-x1)*mu**q1       
        w2 = (s-x2)*mu**q2
        D = Matrix([[w1/N,0],     
                    [0,w2/N]])

        M = Matrix([[ F[0,0]+F[1,1], F[1,0], F[0,1]],
                [ F[0,1] , F[0,0] , 0],
                [ F[1,0] , 0 , F[1,1]]])
        b = Matrix([0, w1/N, w2/N])

        C = (M.subs(parameters).inv())*b.subs(parameters)
        C = Matrix([[C[1], C[0]], [C[0],C[2]]])

        A = exp(-tau*F.transpose().subs(parameters))
        
        L = (2/F.trace())*(F[0,1]*D[1,1] - F[1,0*D[0,0]])
        l1, l2 = F.subs(parameters).eigenvals().keys()
        ctau = ((L/sqrt(F.trace()**2 - 4*F.det()))*(exp(-l2*tau)-exp(-l1*tau))).subs(parameters)

        ang_mo = []
        for t in t_range:
            #ans = C*A.subs({tau:t})
            #ang_mo.append(float(ans[0,1]-ans[1,0]))
            ang_mo.append(float(ctau.subs({tau:t})))
            
        return np.array(t_range), np.array(ang_mo)
    
    

def calculate_matrices(parameters):
    """Returns F,D,C for parameter values"""
    
    s, z, N, tau = symbols('s z N tau')
    q1, q2 = symbols('q_1 q_2')

    if parameters['z'] < 1/6:
        # Lower FP
        x1 = (s/(2*s*(1-s)))*(1-s-sqrt(s*(4-3*s)-1))
        x2 = (s-sqrt(s*(4-3*s)-1))/2
        x1 = simplify(x1.subs(dict(q_1=1,q_2=2)))
        x2 = simplify(x2.subs(dict(q_1=1,q_2=2)))

    else:    
        # Middle FP
        x1 = s/2
        x2 = s/2

    mu = simplify(z + x1 + x2)
    X = { 1:simplify(q1*(s - x1)*(mu**q1)/(mu*(1-mu))),   # X_1mu 
          2:simplify(q2*(s - x2)*(mu**q2)/(mu*(1-mu))) }  # X_2mu

    F = Matrix([[ mu**q1+(1-mu)**q1-X[1], -X[1]],
              [ -X[2], mu**q2+(1-mu)**q2-X[2]]])

    w1 = (s-x1)*mu**q1       
    w2 = (s-x2)*mu**q2
    D = Matrix([[w1/N,0],     
                [0,w2/N]])

    M = Matrix([[ F[0,0]+F[1,1], F[1,0], F[0,1]],
            [ F[0,1] , F[0,0] , 0],
            [ F[1,0] , 0 , F[1,1]]])
    b = Matrix([0, w1/N, w2/N])

    C = (M.subs(parameters).inv())*b.subs(parameters)
    C = Matrix([[C[1], C[0]], [C[0],C[2]]])

    return F,D,C

def calculate_angular_momentum(F,D,C,t_range):
    """ """
    tau = symbols('tau')
    L = (2/F.trace())*(F[0,1]*D[1,1] - F[1,0]*D[0,0])
    l1, l2 = F.subs(parameters).eigenvals().keys()
    ctau = ((L/sqrt(F.trace()**2 - 4*F.det()))*(exp(-l2*tau)-exp(-l1*tau))).subs(parameters)

    ang_mo = []
    for t in t_range:
        ang_mo.append(float(ctau.subs({tau:t})))

    return np.array(t_range), np.array(ang_mo)

def calculate_angular_momentum2(F,D,C,t_range):
    tau = symbols('tau')
    
    A = exp(-tau*F.transpose().subs(parameters))
    ang_mo = []
    for t in t_range:
        ans = C*A.subs({tau:t})
        ang_mo.append(float(ans[0,1]-ans[1,0]))

    return np.array(t_range), np.array(ang_mo)

def calculate_max(F,D):
    
    Delta = sqrt(F.trace()**2 - 4*F.det())
    L = (2/F.trace())*(F[0,1]*D[1,1] - F[1,0]*D[0,0])
    l1, l2 = F.subs(parameters).eigenvals().keys()
    tau_star = log(l1/l2)/Delta
    
    #a_star = L*l1**(-l1/Delta)*l2**(-l2/Delta)/Delta #Wrong
    
    a_star = (L/Delta)*((l2/l1)**(-l1/Delta)-(l2/l1)**(-l2/Delta)) 
    
    a_star = (L/Delta)*(2*F[1,1]-F.trace())*((l2/l1)**(-l1/Delta)*(Delta/l2)) 
    
    a_star = L*l2**(-l2/Delta)*l1**(l2/Delta - 1)
    
    return tau_star.subs(parameters).evalf(), a_star.subs(parameters).evalf()

def compare_simulations(N=1000, z=0.2):
    parameters = dict(s=0.5-z, q_1=1, q_2=2, z=z, N=N)
    F,D,C = calculate_matrices(parameters)
    t,a = calculate_angular_momentum2(F,D,C, arange(20))
    t_s, a_s = calculate_max(F,D)
    
    plt.plot(t,a,'g', label='Analytical')
    plt.axhline(a_s, color='k', linestyle='--')
    plt.axvline(-t_s, color='k', linestyle='--')
    
    data = np.load('0{}z_12_{}_angular_0_10000.npz'.format(int(z*100) if z!=0.01 else '01',N))
    plt.plot(arange(10000)/N,data['ang_mean'], label='Simulation', color='r')
    plt.errorbar(arange(10000)/N,data['ang_mean'], xerr=0, yerr=data['ang_var']**2, alpha=0.01, color='r')
    plt.legend(loc='best')
    plt.xlim((0,20))
    
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    
    plt.xlabel('MC Step')
    plt.ylabel(r'$\langle L_t \rangle/N$')
