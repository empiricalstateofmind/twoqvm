import pandas as pd
from numpy import sqrt, exp
from numpy.linalg import eigvals, matrix_power
from scipy.optimize import brentq

from .functions import *

__all__ = ['TwoQVoterModel', 'InfiniteTwoQVoterModel']

class InfiniteMethods(object):
    """ 
    Abstract container for all methods associated with the infinite model.
    """

    @staticmethod
    def _rho(xi, si, qi):
        """ 
        Calculates the intermediate parameter rho.

        Args:
            xi (float) - density of of positive si agents.
            si (float) - total density of si agents.
            qi (float/int) - associated qi with population of si agents.
        
        Returns:
            rho (float) - intermediate parameter
        """
        return (si / xi - 1) ** (1 / qi)

    @staticmethod
    def _rho_inv(rho, si, qi):
        """ 
        Calculates xi from the parameter rho.

        Args:
            rho (float) - intermediate parameter.
            si (float) - total density of si agents.
            qi (float/int) - associated qi with population of si agents.
        
        Returns:
            xi (float) - density of positive si agents
        """
        return si / (1 + rho ** qi)

    @staticmethod
    def z_critical(q1, q2):
        """
        Calculates the critical value of z in the symmetric case s1=s2.

        Args:
            q1, q2 (int) - susceptible types (number of neighbour queries).

        Returns:
            z_c (float) - the critical value of z.
        """

        return 0.5*(q1 + q2 - 2)/(q1 + q2)

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
        a = self.s[0] / (1 + r ** self.q[0])
        b = self.s[1] / (1 + r ** self.q[1])
        c = 1 / (1 + r)
        return self.z[1] + a + b - c

    def fixed_points(self):
        """
        Calculates the fixed points of the system.

        Returns:
            fps (list) - a list of tuples of the fixed points (x1,x2). 
        """
        fps = []
        eps = 1e-5

        def g(x):
            return self._fixed_point_function(x, 0)

        x1root = eps
        for b in np.linspace(3 * eps, self.s[0], 100):
            try:
                new_x1root = brentq(g, x1root + eps, b)
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
        Note: This will give F at the lower stable fixed point when there are two.
        """
        (x1, x2) = self.fixed_points()[0]
        mu = self.z[1] + x1 + x2
        X1 = self.q[0] * (self.s[0] - x1) * (mu ** self.q[0]) / (mu * (1 - mu))
        X2 = self.q[1] * (self.s[1] - x2) * (mu ** self.q[1]) / (mu * (1 - mu))
        F = np.array([[mu ** self.q[0] + (1 - mu) ** self.q[0] - X1, -X1],
                      [-X2, mu ** self.q[1] + (1 - mu) ** self.q[1] - X2]])
        return F

    @property
    def q_eff(self):
        """
        The effective q (for the qVMZ) for the system.
        """
        return sum([s * q for s, q in zip(self.s, self.q)]) / sum(self.s)

    @property
    def z_c(self):
        delta = (self.s[1] - self.s[0]) / 2
        return 0.5 - 1 / ((1 - delta) * self.q[0] + (1 + delta) * self.q[1])


class FiniteMethods(object):
    """
    Abstract container for all methods associated with the finite model.
    """

    def _simulation_setup(self):
        """ 
        Initialises all the required variables to track the simulation.

        Calculates densities from given integer values. 
        Sets the discrete density increment corresponding to a change in sign.
        Prescribes initial condition and creates list for tracking simulation.
        """

        self.t = 0

        self.s = (self.S[0] / self.N, self.S[1] / self.N)
        self.z = (self.Z[0] / self.N, self.Z[1] / self.N)

        self.x0 = (self.rho[0] * self.s[0], self.rho[1] * self.s[1])
        self.x = self.x0
        self.dx = 1.0 / self.N

        if self.max_iterations is not None:
            self.x1_track = np.zeros(self.max_iterations + 1, dtype=np.uint16)
            self.x1_track[self.t] = self._to_int(self.x0[0])
            self.x2_track = np.zeros(self.max_iterations + 1, dtype=np.uint16)
            self.x2_track[self.t] = self._to_int(self.x0[1])
            self.update = self._array_update
        else:
            self.x1_track = [self._to_int(self.x0[0])]
            self.x2_track = [self._to_int(self.x0[1])]
            self.update = self._append_update
        self._set_transition_matrix()

        return None

    def _to_int(self, density):
        """
        Coverts a density into an integer.

        Args:
            density (float):

        Returns:
            number (int):
        """
        return int(density * self.N + 0.5)

    def load_timeseries(self, filename, var_names=('x1_track', 'x2_track')):
        """
        Loads a time series from file.

        Args:
            filename (str) - filename/path to file.
            var_names (str tuple) - dictionary keys to unpack to x1_track, x2_track.

        Returns:
            None
        """
        data = np.load(filename)
        self.x1_track, self.x2_track = data[var_names[0]], data[var_names[1]]
        return None

    def load_stationary_distribution(self, filename, var_name='X'):
        """
        Loads a stationary distribution from file.

        Args:
            filename (str) - filename/path to file.
            var_name (str) - variable name to unpack X.

        Returns:
            None
        """
        data = np.load(filename)
        self.X = data[var_name]
        return None

    def _append_update(self, xi, val):
        """Updates the simulation history by list update."""
        if xi == 0:
            self.x1_track.append(val)
        elif xi == 1:
            self.x2_track.append(val)
        return None

    def _array_update(self, xi, val):
        """Updates the simulation history by array update."""
        if xi == 0:
            self.x1_track[self.t] = val
        elif xi == 1:
            self.x2_track[self.t] = val
        return None

    def _set_transition_matrix(self):
        """
        Calculates the transition probabilities for use in the simulation.

        Returns:
            None
        """
        mu1 = self.z[0] + self.s[0] + self.s[1] - (self.x[0] + self.x[1])
        mu2 = self.z[1] + self.x[0] + self.x[1]

        self.Tr = [(self.s[0] - self.x[0]) * mu2 ** self.q[0],  # x1 -> x1 + 1
                   (self.x[0]) * mu1 ** self.q[0],  # x1 -> x1 - 1
                   (self.s[1] - self.x[1]) * mu2 ** self.q[1],  # x2 -> x2 + 1
                   (self.x[1]) * mu1 ** self.q[1]]  # x2 -> x2 - 1
        return None

    def run_iteration(self):
        """
        Run one iteration of the model.

        Returns:
            None
        """
        r1 = np.random.random()
        cumsum = np.cumsum(self.Tr)
        ix = np.searchsorted(cumsum, r1)
        self.t += 1

        if ix == 0:
            self.x = (self.x[0] + self.dx, self.x[1])
        elif ix == 1:
            self.x = (self.x[0] - self.dx, self.x[1])
        elif ix == 2:
            self.x = (self.x[0], self.x[1] + self.dx)
        elif ix == 3:
            self.x = (self.x[0], self.x[1] - self.dx)

        self.update(0, self._to_int(self.x[0]))
        self.update(1, self._to_int(self.x[1]))

        self._set_transition_matrix()
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
                print(_i, end='\r')
                self.run_iteration()
        else:
            for _i in range(num_iterations):
                self.run_iteration()
        return None

    @property
    def D(self):
        """
        The linear diffusion matrix D at the stable fixed point.
        Note: If there are two stable fixed points, this will be D around the lower FP.
        """
        (x1, x2) = self.fixed_points()[0]
        mu = self.z[1] + x1 + x2
        w1 = (self.s[0] - x1) * mu ** self.q[0]
        w2 = (self.s[1] - x2) * mu ** self.q[1]
        D = np.array([[w1 / self.N, 0],
                      [0, w2 / self.N]])
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
            t_range (iterable) - the values of t to calculate the lagged two-point correlation.

        Returns:
            t_range, ang_mo (np.array, np.array) - the ts and corresponding correlations.
        """

        F = self.F
        D = self.D

        L = (2 / F.trace()) * (F[0, 1] * D[1, 1] - F[1, 0] * D[0, 0])
        l1, l2 = eigvals(F).real
        delta = sqrt(F.trace() ** 2 - 4 * np.linalg.det(F))
        #ctau = (L / delta)

        ang_mo = []
        for tau in t_range:
            ang_mo.append(L * (exp(-l1 * tau) - exp(-l2 * tau)) / (l1 - l2))

        return np.array(t_range), -np.array(ang_mo)

    def angular_velocity(self, t, sample_size, deviations=True):
        """ 
        Calculates the angular velocity for a given t.

        This is given by the correlation <x1(s)x2(s+t)> - <x1(s+t)x2(s)>.
        The equivalent can be derived for the deviations from the mean.

        Args:
            t (int) - Time difference which correlation is calculated over.
            sample_size (int) - The number of samples used to average over (max = number of iterations - sample_size - t).
            deviations (bool) - If True, use the deviations from the mean rather than true values.

        Note:
            This method will convert xi_track to a numpy array (and possibly subtract the mean which would be irreversible).
        """

        self.x1_track = np.array(self.x1_track)
        self.x2_track = np.array(self.x2_track)
        return angular_velocity(self.x1_track, self.x2_track, t, sample_size, deviations)

    def _generate_transition_matrix(self, sparse=False):
        """
        Generates the transition matrix for the dynamics. 

        Args:
            sparse (bool) - If sparse will use scipy sparse matrix (slow).

        Returns:
            P - the transition matrix.

        Note:
            P is encoded as
                    P((0,0)) P((0,1)) P((0,2)) ...
            P((0,0))
            P((0,1))
            P((0,2))
              ...
        """

        S1, S2 = self.S[0], self.S[1]

        if sparse:
            import scipy.sparse as sp
            P = sp.dok_matrix(((S1 + 1) ** 2, (S2 + 1) ** 2), dtype=np.float)
        else:
            P = np.zeros(shape=((S1 + 1) ** 2, (S2 + 1) ** 2), dtype=np.float)

        for i in range((S1 + 1) ** 2):
            for j in range((S2 + 1) ** 2):
                X1, X2 = self.convert_to_x_pair(i, S1, S2)
                Y1, Y2 = self.convert_to_x_pair(j, S1, S2)
                if X1 + 1 == Y1 and X2 == Y2:
                    P[i, j] = self._w_1_plus(X1, X2)
                elif X1 - 1 == Y1 and X2 == Y2:
                    P[i, j] = self._w_1_minus(X1, X2)
                elif X2 + 1 == Y2 and X1 == Y1:
                    P[i, j] = self._w_2_plus(X1, X2)
                elif X2 - 1 == Y2 and X1 == Y1:
                    P[i, j] = self._w_2_minus(X1, X2)

        for i, val in enumerate(1 - P.sum(axis=1)):
            P[i, i] = val

        P = P / P.sum(axis=1)

        return P

    def calculate_stationary_distribution(self, iterations=2, x=None, filename=False, sparse=False):
        """Calculates the stationary distribution of the model.

        Args:
            iterations (int) - number of iterations of P to calculate (2^iterations).
            x (array) - initial distribution (default: uniform).
            filename (str) - filename/path to save distribution.
            sparse (bool) - if True, use scipy sparse matrices for P (slow).

        Returns:
            X (np.ndarray) - the stationary distribution.
        """

        P = self._generate_transition_matrix(sparse)

        if x is None:
            x = np.ones((1, (self.S[0] + 1) * (self.S[1] + 1)))

        if sparse:
            X = (x * (P ** (2 ** iterations))).reshape((self.S[0] + 1, self.S[1] + 1))
        else:
            X = (x.dot(matrix_power(P, 2 ** iterations))).reshape((self.S[0] + 1, self.S[1] + 1))

        X = X / X.sum()

        if filename:
            if self.S[0] == self.S[1] and self.Z[1] == self.Z[0]:
                np.savez(
                    filename + 'stat_{}N_{}S_{}Z_{}{}.npz'.format(self.N, self.S[0], self.Z[0], self.q[0], self.q[1]),
                    X=X)
            elif self.S[0] == self.S[1]:
                np.savez(filename + 'stat_{}N_{}S_{}Zp_{}Zm_{}{}.npz'.format(self.N, self.S[0], self.Z[1], self.Z[0],
                                                                             self.q[0], self.q[1]), X=X)
            else:
                np.savez(
                    filename + 'stat_{}N_{}S1_{}S2_{}Zp_{}Zm_{}{}.npz'.format(self.N, self.S[0], self.S[1], self.Z[1],
                                                                              self.Z[0], self.q[0], self.q[1]), X=X)

        self.X = X
        return X

    def calculate_stationary_currents(self, iterations, filename=None):
        """
        Calculates the stationary probability currents.

        Args:
            iterations (int) - number of iterations of P to calculate (2^iterations).
            filename (str) - filename/path to load a previously calculated stationary distribution.

        Returns:
            U,V - currents at each point in the configuration space in each direction.
        """

        if filename is not None:
            self.X = np.load(filename)['X']
        elif not hasattr(self, 'X'):
            x = np.ones((self.S[0] + 1, self.S[1] + 1)) / ((self.S[0] + 1) * (self.S[1] + 1))
            self.X = self.calculate_stationary_distribution(iterations, x=x.flatten())[2]

        return np.array(np.fromfunction(self.current, shape=self.X.shape, dtype=int, P=self.X))

    def _w_1_plus(self, X1, X2):
        """Probability current of x1 -> x1 + 1."""
        x1 = X1 / self.N
        x2 = X2 / self.N
        return (self.s[0] - x1) * (self.z[1] + x1 + x2) ** self.q[0]

    def _w_1_minus(self, X1, X2):
        """Probability current of x1 -> x1 - 1."""
        x1 = X1 / self.N
        x2 = X2 / self.N
        return x1 * (1 - self.z[1] - x1 - x2) ** self.q[0]

    def _w_2_plus(self, X1, X2):
        """Probability current of x2 -> x2 + 1."""
        x1 = X1 / self.N
        x2 = X2 / self.N
        return (self.s[1] - x2) * (self.z[1] + x1 + x2) ** self.q[1]

    def _w_2_minus(self, X1, X2):
        """Probability current of x2 -> x2 - 1."""
        x1 = X1 / self.N
        x2 = X2 / self.N
        return x2 * (1 - self.z[1] - x1 - x2) ** self.q[1]

    @staticmethod
    def convert_to_x_pair(i, S1, S2):
        """ Converts an index to the corresponding S in the transition matrix formulation. """
        return i // (S1 + 1), i % (S1 + 1)

    def current(self, X1, X2, P):
        """ Calculates the current at a point X1, X2. """
        # return (-(self._w_1_plus(X1, X2) - self._w_1_minus(X1, X2)) * P[X1, X2],
        #         (self._w_2_plus(X1, X2) - self._w_2_minus(X1, X2)) * P[X1, X2])

        k1 = (self._w_1_plus(X1, X2) - self._w_1_minus(X1, X2)) * P[X1, X2]
        k1 += (X1 >= 1) * (self._w_1_plus(X1 - 1, X2) * P[np.maximum((X1 - 1),0), X2])
        k1 -= (X1 <= self.S[0] - 1) * (self._w_1_minus(X1 + 1, X2) * P[np.minimum((X1 + 1),self.S[0]), X2])

        k2 = (self._w_2_plus(X1, X2) - self._w_2_minus(X1, X2)) * P[X1, X2]
        k2 += (X2 >= 1) * (self._w_2_plus(X1, X2 - 1) * P[X1, np.maximum((X2 - 1),0)])
        k2 -= (X2 <= self.S[1] - 1) * (self._w_2_minus(X1, X2 + 1) * P[X1, np.minimum((X2 + 1),self.S[1])])
        return k1, k2

    def save_timeseries(self, filename=None, compressed=True):
        """
        Saves the time series (and parameter values) to file.

        Args:
            filename (str) - filename/path to save file (without extension).
            compressed (bool) - If True, saves as a .npz file.

        Returns:
            None
        """

        parameters = dict(N=self.N,
                          S1=self.S[0],
                          S2=self.S[1],
                          ZM=self.Z[0],
                          ZP=self.Z[1],
                          q1=self.q[0],
                          q2=self.q[1])

        if filename is None:
            filename = "{N}N_{S1}S1_{S2}S2_{ZM}ZM_{ZP}ZP_{q1}{q2}".format(**parameters)

        if compressed:
            self.x1_track = np.array(self.x1_track, dtype=np.uint16)
            self.x2_track = np.array(self.x2_track, dtype=np.uint16)
            np.savez_compressed("{}.npz".format(filename), x1_track=self.x1_track, x2_track=self.x2_track,
                                parameters=parameters)
        else:
            self.x1_track = np.array(self.x1_track, dtype=np.uint16)
            self.x2_track = np.array(self.x2_track, dtype=np.uint16)

            df = pd.DataFrame({'t': np.arange(len(self.x1_track)), 'x1': self.x1_track, 'x2': self.x2_track},
                              dtype=int)
            df.to_csv("{}.csv".format(filename), index=False)
        return None

    def switching_time(self, xi, lb=None, ub=None, distribution=False):
        """
        Calculates the mean switching time of a species between two values.

        Args:
            xi (binary) - which species to consider the switching of, 0 for x1, 1 for x2.
            lb (float) - lower value.
            ub (float) - upper value.
            distribution (bool) - If True, returns the distribution of switching times.

        Returns:
            time (float) - mean switching time.

        Note:
            1. If either lb or ub are not prescribed then the switching time between the fixed points is calculated.
            2. Some systems have only 1 fixed point so the switching time calculation is unsuitable.
        """

        if lb is None or ub is None:
            fps = self.fixed_points()
            lb, ub = fps[0][xi], fps[-1][xi]

        if distribution:
            if xi == 0:
                return switching_time_dist(self.x1_track, lb, ub)
            elif xi == 1:
                return switching_time_dist(self.x2_track, lb, ub)
            else:
                raise Exception("xi must be 0 (for x1), or 1 (for x2).")
        else:
            if xi == 0:
                return switching_time(self.x1_track, lb, ub)
            elif xi == 1:
                return switching_time(self.x2_track, lb, ub)
            else:
                raise Exception("xi must be 0 (for x1), or 1 (for x2).")

    def switching_periods(self, xi, lb=None, ub=None):
        """
        Returns the indices pairs of periods when the time series is between two values.

        Args:
            xi (binary) - which species to consider the switching of, 0 for x1, 1 for x2.
            lb (float) - lower value.
            ub (float) - upper value.

        Returns:
            swpoints (np.ndarray) - pairs of indices (t1,t2) between which the time series switches between two values.
        """

        if lb is None or ub is None:
            fps = self.fixed_points()
            lb, ub = fps[0][xi], fps[-1][xi]

        if xi == 0:
            return switching_points(self.x1_track, lb, ub)
        elif xi == 1:
            return switching_points(self.x2_track, lb, ub)
        else:
            raise Exception("xi must be 0 (for x1), or 1 (for x2).")


class TwoQVoterModel(FiniteMethods, InfiniteMethods):
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
            max_iterations (int) - The maximum number of iterations that will be run.

        Note: 
            1. For the qVMZ, set S2 = 0. Setting S1 = 0 will not work.
            2. max_iterations solves memory issues and saves computational effect for longer runs.

        Examples:
            1.
                model = TwoQVoterModel(N=100, S=20, Z=30, q=(1,2), rho=0.5)
                model.run_iterations(1000)
                plt.plot(model.x1_track)
                plt.plot(model.x2_track)

            2.
                model = TwoQVoterModel(N=100, S=(15,25), Z=(25,35), q=(1,2), rho=(0,1))
                fps = model.fixed_points()
        """
        self.N = N
        for key, var in zip(['Z', 'S', 'q'], [Z, S, q]):
            if isinstance(var, int) or isinstance(var, np.int64):
                setattr(self, key, (var, var))
            elif (isinstance(var, tuple) or isinstance(var, list)) and len(var) == 2:
                setattr(self, key, var)
            else:
                raise Exception
        assert self.N == sum(self.S) + sum(self.Z)
        assert self.S[0] != 0

        if isinstance(rho, float) or isinstance(rho, int):
            assert (rho >= 0) & (rho <= 1), "Density of +voters needs to be in [0,1]"
            self.rho = (rho, rho)
        elif (isinstance(var, tuple) or isinstance(var, list)) and len(rho) == 2:
            assert (rho[0] >= 0) & (rho[0] <= 1)
            assert (rho[1] >= 0) & (rho[1] <= 1)
            self.rho = rho

        self.max_iterations = None
        for key, val in kwargs.items():
            setattr(self, key, val)

        self._simulation_setup()


class InfiniteTwoQVoterModel(InfiniteMethods):
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

        Examples:
            1.
                model = InfiniteTwoQVoterModel(s=0.3, z=0.2, q=(1,2))
                fps = model.fixed_points()
        """
        for key, var in zip(['z', 's', 'q'], [z, s, q]):
            if isinstance(var, float):
                setattr(self, key, (var, var))
            elif (isinstance(var, tuple) or isinstance(var, list)) and len(var) == 2:
                setattr(self, key, var)
            else:
                raise Exception

        assert np.isclose(1, sum(self.z) + sum(self.s))
