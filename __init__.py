import numpy as np
from numpy.linalg import matrix_power
import matplotlib.pylab as plt
from scipy.optimize import brentq

class TwoQVoterModel(object):

    def __init__(self, **kwargs):
        """
        Define a 2qVM with parameters N, Zp, Zm, S1, S2, q1, q2, rho.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        assert self.N == self.Zp+self.Zm+self.S1+self.S2, "Number of agents must add up to population size"

        if isinstance(self.rho, float) or isinstance(self.rho, int):
            assert (self.rho>=0) & (self.rho<=1), "Density of +voters needs to be in [0,1]"
            self.rho = (self.rho,self.rho)
        elif isinstance(self.rho, tuple) and len(self.rho)==2:
            assert (self.rho[0]>=0) & (self.rho[0]<=1)
            assert (self.rho[1]>=0) & (self.rho[1]<=1)
        
        self.setup()

    def setup(self):
        """ Initialises all the required variables to track the simulation """
        
        self.s1, self.s2 = self.S1/self.N, self.S2/self.N
        self.zp, self.zm = self.Zp/self.N, self.Zm/self.N
            
        self.s = (1 - self.zp - self.zm)/2 
        self.x1, self.x2 = self.rho[0]*self.s1, self.rho[1]*self.s2 
        self.dx = 1.0/self.N
        self.x1_track = [self.x1]
        self.x2_track = [self.x2]
        self.set_transition_matrix()
        self.t = 0

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
    
    def _rho(self, xi, si, qi):
        return (si/xi - 1)**(1/qi)

    def _fixed_point_function(self, xi, s, z, q, species):
        r = self._rho(xi, s[species], q[species])
        a = self.s1/(1+r**self.q1)
        b = self.s2/(1+r**self.q2)
        c = 1/(1+r)
        return z + a + b - c

    def fixed_points(self):
        """Returns the fixed points of the system [(x1,x2,stability)]."""
        fps = []
        eps = 1e-5

        g = lambda x: self._fixed_point_function(x, (self.s1, self.s2), self.zp, (self.q1, self.q2), 0)
        h = lambda x: self._fixed_point_function(x, (self.s1, self.s2), self.zp, (self.q1, self.q2), 1)

        x1root, x2root = eps, eps
        for b in np.linspace(3*eps,self.s1,100):
            try:
                newx1 = brentq(g, x1root+eps, b)
            except ValueError as e:
                continue
            if newx1 > x1root and not np.isclose(newx1, x1root):
                fps.append(newx1)
                x1root = newx1

        # s1_left, s1_right = sorted([(self.s1)*(self.zp/(self.zm+self.zp)),(self.s1)*(self.zm/(self.zm+self.zp))])
        # s2_left, s2_right = sorted([(self.s2)*(self.zp/(self.zm+self.zp)),(self.s2)*(self.zm/(self.zm+self.zp))])

        # for (a,b), (c,d) in zip([(eps,s1_left-eps), (s1_left-eps, s1_right+eps), (s1_right+eps, self.s1-eps)],
        #                         [(eps,s2_left-eps), (s2_left-eps, s2_right+eps), (s2_right+eps, self.s2-eps)]):
        #     try:
        #         print(a,b,c,d)
        #         fps.append((brentq(g, a, b)))#, brentq(h, c, d)))
        #     except ValueError as e:
        #         print(e)
        #         continue # Value error is raised when there is not a root in the interval.

        # fps = np.vstack([fsolve(g, [eps, self.s1/2 - eps, self.s1/2 + eps, self.s1-eps]),
        #                             fsolve(h, [eps, self.s2/2 - eps, self.s2/2 + eps, self.s2-eps])]).T
        # fps = np.vstack({tuple(row) for row in fps})
        # fps = np.sort(fps, axis=0)

        # else:
        #     s1, s2, zp, zm, N, tau = symbols('s_1 s_2 z_p z_m N tau')
        #     q1, q2 = symbols('q_1 q_2')
        #     rho = symbols(r'\rho')
        #     parameters = dict(s_1=self.s1, s_2=self.s2, q_1=self.q1, q_2=self.q2, z_p=self.zp, z_m=self.zm, N=self.N)
            
        #     exp = (1/(1+rho) - zp - s1/(1+rho**q1) - s2/(1+rho**q2)).subs(parameters)
        #     rhos = solve(exp, rho)
            
        #     for val in rhos:
        #         real, im = (float(x) for x in val.as_real_imag())
        #         if np.isclose(im,0):
        #             fps.append((self.s1/(1+real**self.q1), self.s2/(1+real**self.q2)))
        return fps
    

        #if self.s1 == self.s2 and self.zp==self.zm:
        #    s=self.s1
        #    z=self.zp
        #    fp1 = s/2
        #    fps.append((fp1,fp1))
        #    if z < 1/6:
        #        fp2a,fp2b = (s/(2*(1-s)))*(1-s-(s*(4-3*s)-1)**0.5), 0.5*(s-(s*(4-3*s)-1)**0.5)
        #        fp3a,fp3b = (s/(2*(1-s)))*(1-s+(s*(4-3*s)-1)**0.5), 0.5*(s+(s*(4-3*s)-1)**0.5) 
        #        fps.append((fp2a,fp2b))
        #        fps.append((fp3a,fp3b))
        #    return fps
        
        
    def run_iteration(self):
        """ Run one iteration of the model """
        r1 = np.random.random()
        cumsum = np.cumsum(self.Tr)
        ix = np.searchsorted(cumsum, r1)
        
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
        self.t += 1
        return None

    def run_iterations(self, num_iterations, verbose=False):
        assert (num_iterations+self.t <= self.max_iterations), "Increase max_iterations" 
        
        if verbose:
            for _i in range(num_iterations):
                print(_i, end='\r')
                self.run_iteration()
        else:
            for _i in range(num_iterations):
                self.run_iteration()
        return None
    
    def angular_velocity(self, t, T, burn=10000):
        angvec = [self.x1_track[tau]*self.x2_track[tau+t] - self.x1_track[tau+t]*self.x2_track[tau] for tau in range(burn,burn+T-t)] 
        angvec = sum(angvec)/(T-t)
        return angvec

    def save_timeseries(self, filename, compressed=True):
        parameters = dict(N=self.N, zp=self.zp, zm=self.zm, s1=self.s1, s2=self.s2, q1=self.q1, q2=self.q2, rho=self.rho)
        
        if compressed:
            np.savez_compressed("{}.npz".format(filename), x1_track=self.x1_track, x2_track=self.x2_track,
                                parameters=parameters)
            
        else:
            df = pd.DataFrame({'x1':self.x1_track, 'x2':self.x2_track})
            df.to_csv("{}.csv".format(filename), index=False)
        return None 
    
    @property
    def F(self):
        """
        Calculates the linear stability matrix F at the stable fixed point.
        """
        (x1,x2) = self.fixed_points()[0]
        mu = self.zp + x1 + x2
        X1 = self.q1*(self.s1-x1)*(mu**self.q1)/(mu*(1-mu))
        X2 = self.q2*(self.s2-x2)*(mu**self.q2)/(mu*(1-mu))
        F = np.array([[ mu**self.q1 + (1-mu)**self.q1 - X1 , -X1],
                      [ -X2 , mu**self.q2 + (1-mu)**self.q2 - X2]])
        return F

    @property
    def D(self):
        """
        Calculates the linear diffusion matrix D at the stable fixed point.
        """
        (x1,x2) = self.fixed_points()[0]
        mu = self.zp + x1 + x2
        w1 = (self.s1-x1)*mu**self.q1
        w2 = (self.s2-x2)*mu**self.q2
        D = np.array([[w1/self.N, 0],
                      [0, w2/self.N]])
        return D
    
    @property
    def C(self):
        """
        Calculates the correlation matrix C at the stable fixed point.
        """

        F = self.F
        D = self.D
        
        M = np.array([[ F[0,0]+F[1,1], F[1,0], F[0,1]],
                     [ F[0,1] , F[0,0] , 0],
                     [ F[1,0] , 0 , F[1,1]]])
        b = np.array([0, D[0,0], D[1,1]])

        C = np.linalg.inv(M).dot(b)
        C = np.array([[C[1], C[0]], [C[0], C[2]]])      
        return C
    
    def calculate_angular_momentum(self, t_range):
        """
        Calculates the analytical angular momentum in the steady state over a given range.
        Currently restricted to asymmetry zealotry - needs generalising.
        """
        
        F = self.F
        D = self.D

        from scipy.linalg import eigvals
        L = (2/F.trace())*(F[0,1]*D[1,1] - F[1,0]*D[0,0])
        l1, l2 = eigvals(F).astype(float)
        delta = sqrt(F.trace()**2 - 4*np.linalg.det(F))
        ctau = (L/delta)
        
        ang_mo = []
        for tau in t_range:
            ang_mo.append( ctau*(exp(-l1*tau)-exp(-l2*tau))/(l1-l2))
        
        return np.array(t_range), -np.array(ang_mo)

    
    def calculate_angular_momentum_OLD(self,t_range): # This needs work
            
        s, z, N, tau = symbols('s z N tau')
        q1, q2 = symbols('q_1 q_2')
        parameters = dict(s=self.s1, q_1=self.q1, q_2=self.q2, z=self.zp, N=self.N)

        if self.zp < 1/6: # This needs generalising
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
        
        L = (2/F.trace())*(F[0,1]*D[1,1] - F[1,0]*D[0,0])
        l1, l2 = F.subs(parameters).eigenvals().keys()
        ctau = ((L/sqrt(F.trace()**2 - 4*F.det()))*(exp(-l2*tau)-exp(-l1*tau))/(l1-l2)).subs(parameters)

        ang_mo = []
        for t in t_range:
            #ans = C*A.subs({tau:t})
            #ang_mo.append(float(ans[0,1]-ans[1,0]))
            ang_mo.append(float(ctau.subs({tau:t})))
            
        return np.array(t_range), np.array(ang_mo)
    
    def generate_transition_matrix(self, sparse=False):
        """ Generates the transition matrix for the dynamics """
    
        S1, S2 = self.S1, self.S2
        
        if sparse:
            import scipy.sparse as sp
            P = sp.dok_matrix(((S1+1)**2,(S2+1)**2), dtype=np.float)
        else:
            P = np.zeros(shape=((S1+1)**2,(S2+1)**2))
            
        for i in range((S1+1)**2):
            for j in range((S2+1)**2):
                X1,X2 = self.convert_to_x_pair(i, S1, S2)
                Y1,Y2 = self.convert_to_x_pair(j, S1, S2)
                if X1 + 1 == Y1 and X2 == Y2:
                    P[i,j] = self.w_1_plus(X1, X2)
                elif X1 - 1 == Y1 and X2 == Y2:
                    P[i,j] = self.w_1_minus(X1, X2)
                elif X2 + 1 == Y2 and X1 == Y1:
                    P[i,j] = self.w_2_plus(X1, X2)
                elif X2 - 1 == Y2 and X1 == Y1:
                    P[i,j] = self.w_2_minus(X1, X2)
        
        for i,val in enumerate(1 - P.sum(axis=1)):
            P[i,i] = val
            
        return P
    
    def calculate_stationary_distribution(self, iterations=2, x=None, filename=False, sparse=False):
        """ Calculates the stationary distribution of the model. """

        P = self.generate_transition_matrix(sparse)
        
        if x is None:
            x = np.ones((1, (self.S1+1)*(self.S2+1)))

        if sparse:
            X = (x*(P**(2**iterations))).reshape((self.S1+1,self.S2+1))
        else:
            X = (x.dot(matrix_power(P, 2**iterations))).reshape((self.S1+1,self.S2+1))
        
        X = X/X.sum()
                                       
        x1_dist = X.sum(axis=1)
        x2_dist = X.sum(axis=0).T
              
        if filename:
            if self.S1==self.S2 and self.Zp == self.Zm:
                np.savez(filename+'stat_{}N_{}S_{}Z_{}{}.npz'.format(self.N, self.S1, self.Zm, self.q1, self.q2), X=X)
            elif self.S1==self.S2:
                np.savez(filename+'stat_{}N_{}S_{}Zp_{}Zm_{}{}.npz'.format(self.N, self.S1, self.Zp, self.Zm, self.q1, self.q2), X=X)
            else:
                np.savez(filename+'stat_{}N_{}S1_{}S2_{}Zp_{}Zm_{}{}.npz'.format(self.N, self.S1, self.S2, self.Zp, self.Zm, self.q1, self.q2), X=X)

        self.X = X
        return x1_dist, x2_dist, X

    def calculate_stationary_currents(self, iterations, filename=None):
        """"""

        if filename is not None:
            self.X = np.load(filename)['X']
        elif not hasattr(self, 'X'):
            x = np.ones((self.S1+1, self.S2+1))/((self.S1+1)*(self.S2+1))
            self.X = self.calculate_stationary_distribution(iterations, x=x.flatten())[2]

        return np.array(np.fromfunction(self.current, shape=self.X.shape, dtype=int, P=self.X))
 
    def w_1_plus(self, X1, X2):
        x1=X1/self.N
        x2=X2/self.N
        return (self.s1 - x1)*(self.zp + x1 + x2)**self.q1

    def w_1_minus(self, X1, X2):
        x1=X1/self.N
        x2=X2/self.N
        return (x1)*(1 - self.zp - x1 - x2)**self.q1

    def w_2_plus(self, X1, X2):
        x1=X1/self.N
        x2=X2/self.N
        return (self.s2 - x2)*(self.zp + x1 + x2)**self.q2

    def w_2_minus(self, X1, X2):
        x1=X1/self.N
        x2=X2/self.N
        return (x2)*(1 - self.zp - x1 - x2)**self.q2

    def convert_to_x_pair(self, i, S1, S2):
        return i//(S1+1),i%(S1+1) 

    def current(self, X1, X2, P):
        return (-(self.w_1_plus(X1, X2) - self.w_1_minus(X1, X2))*P[X1,X2], 
                (self.w_2_plus(X1, X2) - self.w_2_minus(X1, X2))*P[X1,X2])

### END CLASS ###

class InfiniteTwoQVoterModel(TwoQVoterModel):

    def __init__(self, s1, s2, zp, zm, q1, q2):

        self.s1 = s1
        self.s2 = s2
        self.zp = zp
        self.zm = zm
        self.q1 = q1
        self.q2 = q2
        return None

# def calculate_matrices(parameters):
#     """Returns F,D,C for parameter values"""
    
#     s, z, N, tau = symbols('s z N tau')
#     q1, q2 = symbols('q_1 q_2')

#     if parameters['z'] < 1/6:
#         # Lower FP
#         x1 = (s/(2*s*(1-s)))*(1-s-sqrt(s*(4-3*s)-1))
#         x2 = (s-sqrt(s*(4-3*s)-1))/2
#         x1 = simplify(x1.subs(dict(q_1=1,q_2=2)))
#         x2 = simplify(x2.subs(dict(q_1=1,q_2=2)))

#     else:    
#         # Middle FP
#         x1 = s/2
#         x2 = s/2

#     mu = simplify(z + x1 + x2)
#     X = { 1:simplify(q1*(s - x1)*(mu**q1)/(mu*(1-mu))),   # X_1mu 
#           2:simplify(q2*(s - x2)*(mu**q2)/(mu*(1-mu))) }  # X_2mu

#     F = Matrix([[ mu**q1+(1-mu)**q1-X[1], -X[1]],
#               [ -X[2], mu**q2+(1-mu)**q2-X[2]]])

#     w1 = (s-x1)*mu**q1       
#     w2 = (s-x2)*mu**q2
#     D = Matrix([[w1/N,0],     
#                 [0,w2/N]])

#     M = Matrix([[ F[0,0]+F[1,1], F[1,0], F[0,1]],
#             [ F[0,1] , F[0,0] , 0],
#             [ F[1,0] , 0 , F[1,1]]])
#     b = Matrix([0, w1/N, w2/N])

#     C = (M.subs(parameters).inv())*b.subs(parameters)
#     C = Matrix([[C[1], C[0]], [C[0],C[2]]])

#     return F,D,C

# def calculate_angular_momentum(F,D,C,t_range, parameters):
#     """ """
#     tau = symbols('tau')
#     L = (2/F.trace())*(F[0,1]*D[1,1] - F[1,0]*D[0,0])
#     l1, l2 = F.subs(parameters).eigenvals().keys()
#     ctau = ((L/sqrt(F.trace()**2 - 4*F.det()))*(exp(-l2*tau)-exp(-l1*tau))).subs(parameters)

#     ang_mo = []
#     for t in t_range:
#         ang_mo.append(float(ctau.subs({tau:t})))

#     return np.array(t_range), np.array(ang_mo)

# def calculate_angular_momentum2(F,D,C,t_range, parameters):
#     tau = symbols('tau')
    
#     A = exp(-tau*F.transpose().subs(parameters))
#     ang_mo = []
#     for t in t_range:
#         ans = C*A.subs({tau:t})
#         ang_mo.append(float(ans[0,1]-ans[1,0]))

#     return np.array(t_range), np.array(ang_mo)

# def calculate_max(F,D):
    
#     Delta = sqrt(F.trace()**2 - 4*F.det())
#     L = (2/F.trace())*(F[0,1]*D[1,1] - F[1,0]*D[0,0])
#     l1, l2 = F.subs(parameters).eigenvals().keys()
#     tau_star = log(l1/l2)/Delta
    
#     #a_star = L*l1**(-l1/Delta)*l2**(-l2/Delta)/Delta #Wrong
    
#     a_star = (L/Delta)*((l2/l1)**(-l1/Delta)-(l2/l1)**(-l2/Delta)) 
    
#     a_star = (L/Delta)*(2*F[1,1]-F.trace())*((l2/l1)**(-l1/Delta)*(Delta/l2)) 
    
#     a_star = L*l2**(-l2/Delta)*l1**(l2/Delta - 1)
    
#     return tau_star.subs(parameters).evalf(), a_star.subs(parameters).evalf()

# def compare_simulations(N=1000, z=0.2):
#     parameters = dict(s=0.5-z, q_1=1, q_2=2, z=z, N=N)
#     F,D,C = calculate_matrices(parameters)
#     t,a = calculate_angular_momentum2(F,D,C, arange(20))
#     t_s, a_s = calculate_max(F,D)
    
#     plt.plot(t,a,'g', label='Analytical')
#     plt.axhline(a_s, color='k', linestyle='--')
#     plt.axvline(-t_s, color='k', linestyle='--')
    
#     data = np.load('0{}z_12_{}_angular_0_10000.npz'.format(int(z*100) if z!=0.01 else '01',N))
#     plt.plot(arange(10000)/N,data['ang_mean'], label='Simulation', color='r')
#     plt.errorbar(arange(10000)/N,data['ang_mean'], xerr=0, yerr=data['ang_var']**2, alpha=0.01, color='r')
#     plt.legend(loc='best')
#     plt.xlim((0,20))
    
#     plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    
#     plt.xlabel('MC Step')
#     plt.ylabel(r'$\langle L_t \rangle/N$')
    
    
# ## To be merged into the class


# def calculate_stationary_distribution(N=100, S=40, Z=None, q1=1, q2=2, iterations=10000, x=None, save=False):
#     import scipy.sparse as sp
    
#     s = S/N 
#     Z = (N-2*S)/2
#     z = 0.5-s

#     P = sp.dok_matrix(((S+1)**2,(S+1)**2), dtype=np.float)
#     for i in range((S+1)**2):
#         for j in range((S+1)**2):
#             X1,X2 = convert_to_x_pair(i,S)
#             Y1,Y2 = convert_to_x_pair(j,S)
#             if X1 + 1 == Y1 and X2 == Y2:
#                 P[i,j] = w_1_plus(X1, X2, N, s, z, q)
#             elif X1 - 1 == Y1 and X2 == Y2:
#                 P[i,j] = w_1_minus(X1, X2, N, s, z, q)
#             elif X2 + 1 == Y2 and X1 == Y1:
#                 P[i,j] = w_2_plus(X1, X2, N, s, z, p)
#             elif X2 - 1 == Y2 and X1 == Y1:
#                 P[i,j] = w_2_minus(X1, X2, N, s, z, p)

#     for i,val in enumerate(1 - P.sum(axis=1)):
#         P[i,i] = val

#     if x is None:
#         x = np.ones((1, (S+1)**2))/(S+1)**2
        
#     X = (x*(P**iterations)).reshape((S+1,S+1))
    
#     x1_dist = X.sum(axis=1)
#     x2_dist = X.sum(axis=0).T
    
#     if save:
#         np.savez('./stationary_S{}_N{}.npz'.format(S,N), X=X, P=P)
    
#     return x1_dist, x2_dist, X

# def plot_stationary_distribution(x1, x2, S, N, save=False, **kwargs):
#     fig = plt.figure(**kwargs)
#     ax = fig.add_subplot(111)
#     ax.plot(np.linspace(0,1,S+1), x1, label=r'$q=2$')
#     ax.plot(np.linspace(0,1,S+1), x2, label=r'$p=1$')
#     ax.set_xlabel('Proportion of Positive Voters')
#     ax.set_ylabel('Stationary Probability')
#     ax.legend(loc='best')
#     ax.set_title(r'$s = {}, z={}$'.format(S/N, (N//2 - S)/N))
    
#     if save:
#         fig.savefig('stationary_S{}_N{}.png'.format(S,N))

# def w_1_plus(x1,x2,N, s, z, q):
#     x1=x1/N
#     x2=x2/N
#     return (s-x1)*(z+x1+x2)**q

# def w_1_minus(x1,x2,N, s, z, q):
#     x1=x1/N
#     x2=x2/N
#     return (x1)*(z+2*s-x1-x2)**q

# def w_2_plus(x1,x2,N, s, z, p):
#     x1=x1/N
#     x2=x2/N
#     return (s-x2)*(z+x1+x2)**p

# def w_2_minus(x1,x2,N, s, z, p):
#     x1=x1/N
#     x2=x2/N
#     return (x2)*(z+2*s-x1-x2)**p

# def convert_to_x_pair(i,S):
#     return i//(S+1),i%(S+1)