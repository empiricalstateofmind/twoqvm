import numpy as np
from numpy import sqrt, exp
from scipy.optimize import brentq
from scipy.linalg import eigvals

# TO DO
# 1. Add ability to fix the number of iterations (reduce append load when doing large sims)
# 2. Add the stationary distribution functions
# 3. Add helper functions - to CSV e.t.c
# 4. TESTS


class MeanfieldMethods(object):
    """ 
    Abstract container for all methods associated with the meanfield. 

    Methods:
        fixed_points() - Returns the fixed points of the system.

    Properties:
        F - The linear stability matrix about the stable fixed point.

    """

    def _rho(self, xi, si, qi):
        """ 
        Calculates the intermediate parameter rho.

        Args:
            xi (float) - 
            si (float) -
            qi (float/int) - 
        
        Returns:
            rho (float) -
        """
        return (si/xi - 1)**(1/qi)

    def _rho_inv(self, rho, si, qi):
        """ 
        Calculates xi from the parameter rho.

        Args:
            rho (float) - 
            si (float) -
            qi (float/int) - 
        
        Returns:
            xi (float) -
        """
        return si/(1+rho**qi)

    def _fixed_point_function(self, xi, species):
        """
        The fixed point function for xi. Zeros of this function are the fixed points for xi.

        Args:
            xi (float) - Dummy parameter for densities xi.
            species (binary) - 0 for x1, 1 for x2.

        Returns:
            f (float) - function evaluated at xi.
        """
        r = self._rho(xi, self.s[species], self.q[species])
        a = self.s[0]/(1+r**self.q[0])
        b = self.s[1]/(1+r**self.q[1])
        c = 1/(1+r)
        return self.z[1] + a + b - c

    def fixed_points(self):
        """
        Calculates the fixed points of the system.
        
        Args: 
            None

        Returns:
            fps (list) - a list of tuples of the fixed points (x1,x2). 
        """
        fps = []
        eps = 1e-5

        g = lambda x: self._fixed_point_function(x, 0)

        x1root = eps
        for b in np.linspace(3*eps, self.s[0], 100):
            try:
                new_x1root = brentq(g, x1root+eps, b)
            except ValueError as e:
                continue
            else:
                if new_x1root > x1root and not np.isclose(new_x1root, x1root):
                    fps.append(new_x1root)
                    x1root = new_x1root

        if self.s[1] != 0:
            fps = [(x1, self._rho_inv(self._rho(x1, self.s[0], self.q[0]), self.s[1], self.q[1])) for x1 in fps]
        return fps

    @property
    def F(self):
        """
        The linear stability matrix F at the stable fixed point.
        """
        (x1, x2) = self.fixed_points()[0]
        mu = self.z[1] + x1 + x2
        X1 = self.q[0]*(self.s[0]-x1)*(mu**self.q[0])/(mu*(1-mu))
        X2 = self.q[1]*(self.s[1]-x2)*(mu**self.q[1])/(mu*(1-mu))
        F = np.array([[mu**self.q[0] + (1-mu)**self.q[0] - X1, -X1],
                      [-X2, mu**self.q[1] + (1-mu)**self.q[1] - X2]])
        return F


class FiniteMethods(object):
    """ Abstract container for all methods associated with the finite model. """

    def _simulation_setup(self):
        """ 
        Initialises all the required variables to track the simulation.

        Calculates densities from given integer values. 
        Sets the discrete density increment corresponding to a change in sign.
        Prescribes initial condition and creates list for tracking simulation.

        Args:
            None

        Returns:
            None
        """
        
        self.s = (self.S[0]/self.N, self.S[1]/self.N)
        self.z = (self.Z[0]/self.N, self.Z[1]/self.N)
            
        self.x = (self.rho[0]*self.s[0], self.rho[1]*self.s[1]) 
        self.dx = 1.0/self.N
        self.x1_track = [self.x[0]]
        self.x2_track = [self.x[1]]
        self._set_transition_matrix()
        self.t = 0
        return None

    def _set_transition_matrix(self):
        """
        Calculates the transition probabilities for use in the simulation.

        Args:
            None

        Returns:
            None
        """

        self.x = (self.x1_track[-1], self.x2_track[-1])

        mu1 = self.z[0]+self.s[0]+self.s[1]-(self.x[0]+self.x[1])
        mu2 = self.z[1]+self.x[0]+self.x[1]

        self.Tr = [(self.s[0]-self.x[0])*(mu2)**self.q[0],       # x1 -> x1 + 1
                   (self.x[0])*(mu1)**self.q[0],                 # x1 -> x1 - 1
                   (self.s[1]-self.x[1])*(mu2)**self.q[1],       # x2 -> x2 + 1
                   (self.x[1])*(mu1)**self.q[1]]                 # x2 -> x2 - 1
        return None

    def run_iteration(self):
        """
        Run one iteration of the model.

        Args:
            None

        Returns:
            None
        """
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
        
        self._set_transition_matrix()
        self.t += 1
        return None

    def run_iterations(self, num_iterations, verbose=False):
        """
        Runs multiple iterations of the model.

        Args:
            num_iterations (int) - The number of iterations to run. 
            verbose (bool) - If True, prints the iteration number as the simulation runs.

        Returns:
            None
        """
        if verbose:
            for _i in range(num_iterations):
                print(_i, end="\r")
                self.run_iteration()
        else:
            for _i in range(num_iterations):
                self.run_iteration()
        return None

    @property
    def D(self):
        """
        The linear diffusion matrix D at the stable fixed point.
        """
        (x1, x2) = self.fixed_points()[0]
        mu = self.z[1] + x1 + x2
        w1 = (self.s[0]-x1)*mu**self.q[0]
        w2 = (self.s[1]-x2)*mu**self.q[1]
        D = np.array([[w1/self.N, 0],
                      [0, w2/self.N]])
        return D
    
    @property
    def C(self):
        """
        The correlation matrix C at the stable fixed point.
        """

        F = self.F
        D = self.D
        
        M = np.array([[F[0, 0] + F[1, 1], F[1, 0], F[0, 1]],
                     [F[0, 1], F[0, 0], 0],
                     [F[1, 0], 0, F[1, 1]]])
        b = np.array([0, D[0, 0], D[1, 1]])

        C = np.linalg.inv(M).dot(b)
        C = np.array([[C[1], C[0]], [C[0], C[2]]])      
        return C
    
    def calculate_angular_momentum(self, t_range):
        """
        Calculates the analytical angular momentum in the steady state over a given range.
        
        Currently restricted to symmetric zealotry - needs generalising.

        Args:
            t_range (iterable) - 

        Returns:
            t_range, ang_mo (np.array, np.array) - 
        """
        
        F = self.F
        D = self.D

        L = (2/F.trace())*(F[0, 1]*D[1, 1] - F[1, 0]*D[0, 0])
        l1, l2 = eigvals(F).real
        delta = sqrt(F.trace()**2 - 4*np.linalg.det(F))
        ctau = (L/delta)
        
        ang_mo = []
        for tau in t_range:
            ang_mo.append(ctau*(exp(-l1*tau)-exp(-l2*tau))/(l1-l2))
        
        return np.array(t_range), -np.array(ang_mo)


class TwoQVoterModel(FiniteMethods, MeanfieldMethods):
    """ A class for simulation and analysis of the finite 2qVMZ. """

    def __init__(self, N, Z, S, q, rho, *args, **kwargs):
        """ 
        Initialises a finite version of the 2qZM. 

        Args:
            N (int) - The number of agents/nodes in the system
            Z (int/tuple) - The number of negative and positive zealots respectively.
                            If an int is provided the simulation assumes Zp=Zm.
            S (int/tuple) - The number of q1 and q2 susceptibles respectively.
                            if an int is provided the simulation assumes S1=S2.
            q (tuple) - The (q1,q2) agent confidence parameters.
            rho (float/tuple) - The initial fraction of positive susceptibles for S1, S2 respectively.
                                If a float is provided then this density is applied across both groups.

        Note: 
            For the qVMZ, set S2 = 0. Setting S1 = 0 will not work.
        """
        self.N = N
        for key, var in zip(['Z', 'S', 'q'], [Z, S, q]):
            if isinstance(var, int):
                setattr(self, key, (var, var))
            elif len(var) == 2:
                setattr(self, key, var)
            else:
                raise Exception
        assert self.N == sum(self.S) + sum(self.Z)
        assert self.S[0] != 0

        if isinstance(rho, float) or isinstance(rho, int):
            assert (rho >= 0) & (rho <= 1), "Density of +voters needs to be in [0,1]"
            self.rho = (rho, rho)
        elif len(rho) == 2:
            assert (rho[0] >= 0) & (rho[0] <= 1)
            assert (rho[1] >= 0) & (rho[1] <= 1) 
            self.rho = rho

        self._simulation_setup()
        return None


class MeanfieldTwoQVoterModel(MeanfieldMethods):
    """ A class for simulation and analysis of the infinite 2qVMZ. """

    def __init__(self, z, s, q):
        """
        Initialises an infinite version of the 2qZM.

        Args:
            z (float/tuple) - The density of negative and positive zealots respectively.
                            If an int is provided the simulation assumes zp=zm.
            s (float/tuple) - The density of q1 and q2 susceptibles respectively.
                            if an int is provided the simulation assumes S1=S2.
            q (tuple) - The (q1,q2) agent confidence parameters.

        Note:
            For the qVMZ, set s2 = 0. Setting s1 = 0 will not work.
        """
        for key, var in zip(['z', 's', 'q'], [z, s, q]):
            if isinstance(var, float):
                setattr(self, key, (var, var))
            elif len(var,) == 2:
                setattr(self, key, var)
            else:
                raise Exception

        assert 1 == sum(self.z) + sum(self.s)
        return None

                
            #     def angular_velocity(self, t, T, burn=10000):
            #         angvec = [self.x1_track[tau]*self.x2_track[tau+t] - self.x1_track[tau+t]*self.x2_track[tau] for tau in range(burn,burn+T-t)] 
            #         angvec = sum(angvec)/(T-t)
            #         return angvec

            #     def save_timeseries(self, filename, compressed=True):
            #         parameters = dict(N=self.N, zp=self.zp, zm=self.zm, s1=self.s1, s2=self.s2, q1=self.q1, q2=self.q2, rho=self.rho)
                    
            #         if compressed:
            #             np.savez_compressed("{}.npz".format(filename), x1_track=self.x1_track, x2_track=self.x2_track,
            #                                 parameters=parameters)
                        
            #         else:
            #             df = pd.DataFrame({'x1':self.x1_track, 'x2':self.x2_track})
            #             df.to_csv("{}.csv".format(filename), index=False)
            #         return None 
                
            #     def generate_transition_matrix(self, sparse=False):
            #         """ Generates the transition matrix for the dynamics """
                
            #         S1, S2 = self.S1, self.S2
                    
            #         if sparse:
            #             import scipy.sparse as sp
            #             P = sp.dok_matrix(((S1+1)**2,(S2+1)**2), dtype=np.float)
            #         else:
            #             P = np.zeros(shape=((S1+1)**2,(S2+1)**2))
                        
            #         for i in range((S1+1)**2):
            #             for j in range((S2+1)**2):
            #                 X1,X2 = self.convert_to_x_pair(i, S1, S2)
            #                 Y1,Y2 = self.convert_to_x_pair(j, S1, S2)
            #                 if X1 + 1 == Y1 and X2 == Y2:
            #                     P[i,j] = self.w_1_plus(X1, X2)
            #                 elif X1 - 1 == Y1 and X2 == Y2:
            #                     P[i,j] = self.w_1_minus(X1, X2)
            #                 elif X2 + 1 == Y2 and X1 == Y1:
            #                     P[i,j] = self.w_2_plus(X1, X2)
            #                 elif X2 - 1 == Y2 and X1 == Y1:
            #                     P[i,j] = self.w_2_minus(X1, X2)
                    
            #         for i,val in enumerate(1 - P.sum(axis=1)):
            #             P[i,i] = val
                        
            #         return P
                
            #     def calculate_stationary_distribution(self, iterations=2, x=None, filename=False, sparse=False):
            #         """ Calculates the stationary distribution of the model. """

            #         P = self.generate_transition_matrix(sparse)
                    
            #         if x is None:
            #             x = np.ones((1, (self.S1+1)*(self.S2+1)))

            #         if sparse:
            #             X = (x*(P**(2**iterations))).reshape((self.S1+1,self.S2+1))
            #         else:
            #             X = (x.dot(matrix_power(P, 2**iterations))).reshape((self.S1+1,self.S2+1))
                    
            #         X = X/X.sum()
                                                   
            #         x1_dist = X.sum(axis=1)
            #         x2_dist = X.sum(axis=0).T
                          
            #         if filename:
            #             if self.S1==self.S2 and self.Zp == self.Zm:
            #                 np.savez(filename+'stat_{}N_{}S_{}Z_{}{}.npz'.format(self.N, self.S1, self.Zm, self.q1, self.q2), X=X)
            #             elif self.S1==self.S2:
            #                 np.savez(filename+'stat_{}N_{}S_{}Zp_{}Zm_{}{}.npz'.format(self.N, self.S1, self.Zp, self.Zm, self.q1, self.q2), X=X)
            #             else:
            #                 np.savez(filename+'stat_{}N_{}S1_{}S2_{}Zp_{}Zm_{}{}.npz'.format(self.N, self.S1, self.S2, self.Zp, self.Zm, self.q1, self.q2), X=X)

            #         self.X = X
            #         return x1_dist, x2_dist, X

            #     def calculate_stationary_currents(self, iterations, filename=None):
            #         """"""

            #         if filename is not None:
            #             self.X = np.load(filename)['X']
            #         elif not hasattr(self, 'X'):
            #             x = np.ones((self.S1+1, self.S2+1))/((self.S1+1)*(self.S2+1))
            #             self.X = self.calculate_stationary_distribution(iterations, x=x.flatten())[2]

            #         return np.array(np.fromfunction(self.current, shape=self.X.shape, dtype=int, P=self.X))
             
            #     def w_1_plus(self, X1, X2):
            #         x1=X1/self.N
            #         x2=X2/self.N
            #         return (self.s1 - x1)*(self.zp + x1 + x2)**self.q1

            #     def w_1_minus(self, X1, X2):
            #         x1=X1/self.N
            #         x2=X2/self.N
            #         return (x1)*(1 - self.zp - x1 - x2)**self.q1

            #     def w_2_plus(self, X1, X2):
            #         x1=X1/self.N
            #         x2=X2/self.N
            #         return (self.s2 - x2)*(self.zp + x1 + x2)**self.q2

            #     def w_2_minus(self, X1, X2):
            #         x1=X1/self.N
            #         x2=X2/self.N
            #         return (x2)*(1 - self.zp - x1 - x2)**self.q2

            #     def convert_to_x_pair(self, i, S1, S2):
            #         return i//(S1+1),i%(S1+1) 

            #     def current(self, X1, X2, P):
            #         return (-(self.w_1_plus(X1, X2) - self.w_1_minus(X1, X2))*P[X1,X2], 
            #                 (self.w_2_plus(X1, X2) - self.w_2_minus(X1, X2))*P[X1,X2])
