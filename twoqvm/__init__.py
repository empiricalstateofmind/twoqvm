import numpy as np
import pandas as pd
from numpy import sqrt, exp
from numpy.linalg import eigvals, matrix_power
from scipy.optimize import brentq


# TO DO
# 1. TESTS
# 2. Switching Dynamics Theory
# 3. Extract out array methods (angular momentum, switching, e.t.c)
# 4. Add rescaling methods (x1*=x2*, and x1/s1, x2/s2)


class InfiniteMethods(object):
    """ 
    Abstract container for all methods associated with the infinite model. 

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
        return (si / xi - 1) ** (1 / qi)

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
        return si / (1 + rho ** qi)

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
        
        Args: 
            None

        Returns:
            fps (list) - a list of tuples of the fixed points (x1,x2). 
        """
        fps = []
        eps = 1e-5

        g = lambda x: self._fixed_point_function(x, 0)

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

        Args:
            None

        Returns:
            None
        """

        self.t = 0

        self.s = (self.S[0] / self.N, self.S[1] / self.N)
        self.z = (self.Z[0] / self.N, self.Z[1] / self.N)

        self.x = (self.rho[0] * self.s[0], self.rho[1] * self.s[1])
        self.dx = 1.0 / self.N

        if self.max_iterations is not None:
            self.x1_track = np.zeros(self.max_iterations + 1, dtype=float)
            self.x1_track[self.t] = self.x[0]
            self.x2_track = np.zeros(self.max_iterations + 1, dtype=float)
            self.x1_track[self.t] = self.x[1]
            self.update = self._array_update
        else:
            self.x1_track = [self.x[0]]
            self.x2_track = [self.x[1]]
            self.update = self._append_update
        self._set_transition_matrix()

        return None

    def load_timeseries(self, filename, var_names=('x1_track', 'x2_track')):
        """
        Loads a timeseries from file.

        Args:
            filename (str) - 
            var_names (str tuple) -

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
            filename (str) - 
            var_name (str) -

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

        Args:
            None

        Returns:
            None
        """
        self.x = (self.x1_track[self.t], self.x2_track[self.t])

        mu1 = self.z[0] + self.s[0] + self.s[1] - (self.x[0] + self.x[1])
        mu2 = self.z[1] + self.x[0] + self.x[1]

        self.Tr = [(self.s[0] - self.x[0]) * (mu2) ** self.q[0],  # x1 -> x1 + 1
                   (self.x[0]) * (mu1) ** self.q[0],  # x1 -> x1 - 1
                   (self.s[1] - self.x[1]) * (mu2) ** self.q[1],  # x2 -> x2 + 1
                   (self.x[1]) * (mu1) ** self.q[1]]  # x2 -> x2 - 1
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
        self.t += 1

        if ix == 0:
            self.update(0, self.x1_track[self.t - 1] + self.dx)
            self.update(1, self.x2_track[self.t - 1])
        elif ix == 1:
            self.update(0, self.x1_track[self.t - 1] - self.dx)
            self.update(1, self.x2_track[self.t - 1])
        elif ix == 2:
            self.update(0, self.x1_track[self.t - 1])
            self.update(1, self.x2_track[self.t - 1] + self.dx)
        elif ix == 3:
            self.update(0, self.x1_track[self.t - 1])
            self.update(1, self.x2_track[self.t - 1] - self.dx)
        else:
            self.update(0, self.x1_track[self.t - 1])
            self.update(1, self.x2_track[self.t - 1])

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
            t_range (iterable) - 

        Returns:
            t_range, ang_mo (np.array, np.array) - 
        """

        F = self.F
        D = self.D

        L = (2 / F.trace()) * (F[0, 1] * D[1, 1] - F[1, 0] * D[0, 0])
        l1, l2 = eigvals(F).real
        delta = sqrt(F.trace() ** 2 - 4 * np.linalg.det(F))
        ctau = (L / delta)

        ang_mo = []
        for tau in t_range:
            ang_mo.append(ctau * (exp(-l1 * tau) - exp(-l2 * tau)) / (l1 - l2))

        return np.array(t_range), -np.array(ang_mo)

    def angular_velocity(self, t, sample_size, deviations=True):
        """ 
        Calculates the angular velocity for a given t.

        This is given by the correlation <x1(s)x2(s+t)> - <x1(s+t)x2(s)>.
        The equivalent can be derived for the deviations from the mean.

        Args:
            t (int) - Time difference which correlation is calculated over.
            sample_size - The number of samples used to average over (max = number of iterations - sample_size - t).
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
            sparse (bool) -

        Returns:
            P - 
        """

        S1, S2 = self.S[0], self.S[1]

        if sparse:
            import scipy.sparse as sp
            P = sp.dok_matrix(((S1 + 1) ** 2, (S2 + 1) ** 2), dtype=np.float)
        else:
            P = np.zeros(shape=((S1 + 1) ** 2, (S2 + 1) ** 2), dtype=np.float)

        # P is encoded as 
        #           P((0,0)) P((0,1)) P((0,2)) ... 
        # P((0,0))
        # P((0,1))
        # P((0,2))
        #  ...

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
            iterations (int) - 
            x - 
            filename - 
            sparse - 

        Returns:
            X
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
            iterations -
            filename - 

        Returns:
            C -
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
        return (x1) * (1 - self.z[1] - x1 - x2) ** self.q[0]

    def _w_2_plus(self, X1, X2):
        """Probability current of x2 -> x2 + 1."""
        x1 = X1 / self.N
        x2 = X2 / self.N
        return (self.s[1] - x2) * (self.z[1] + x1 + x2) ** self.q[1]

    def _w_2_minus(self, X1, X2):
        """Probability current of x2 -> x2 - 1."""
        x1 = X1 / self.N
        x2 = X2 / self.N
        return (x2) * (1 - self.z[1] - x1 - x2) ** self.q[1]

    def convert_to_x_pair(self, i, S1, S2):
        """ """
        return i // (S1 + 1), i % (S1 + 1)

    def current(self, X1, X2, P):
        """ """
        return (-(self._w_1_plus(X1, X2) - self._w_1_minus(X1, X2)) * P[X1, X2],
                (self._w_2_plus(X1, X2) - self._w_2_minus(X1, X2)) * P[X1, X2])

    def save_timeseries(self, filename=None, compressed=True):
        """
        Saves the timeseries (and parameter values) to file.

        Args:
            filename (str) - 
            compressed (bool) -

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
            np.savez_compressed("{}.npz".format(filename), x1_track=self.x1_track, x2_track=self.x2_track,
                                parameters=parameters)
        else:
            self.x1_track = np.array(self.x1_track) * self.N + 0.5
            self.x2_track = np.array(self.x2_track) * self.N + 0.5

            df = pd.DataFrame({'t': np.arange(len(self.x1_track)), 'x1': self.x1_track, 'x2': self.x2_track}, dtype=int)
            df.to_csv("{}.csv".format(filename), index=False)
        return None

    def switching_time(self, xi, lb=None, ub=None, distribution=False):
        """
        Calculates the mean switching time of a species between two values.

        Args:
            xi (binary) - which species to consider the switching of, 0 for x1, 1 for x2.
            lb (float) - lower value
            up (float) - upper value 

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
        Returns the indices pairs of periods when the timeseries is between two values.

        Args:
            xi (binary) - which species to consider the switching of, 0 for x1, 1 for x2.
            lb (float) - lower value
            up (float) - upper value 

        Returns:
            swpoints (np.ndarray)
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
        return None


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
        """
        for key, var in zip(['z', 's', 'q'], [z, s, q]):
            if isinstance(var, float):
                setattr(self, key, (var, var))
            elif (isinstance(var, tuple) or isinstance(var, list)) and len(var) == 2:
                setattr(self, key, var)
            else:
                raise Exception

        assert np.isclose(1, sum(self.z) + sum(self.s))
        return None


def switching_time(arr, a1, a2):
    """
    Calculates the mean switching time of an array between two values.

    Measures the time taken, on average, for the process to move from the lower value
    to the upper.

    Args:
        arr (np.array) -
        a1 (float) - lower value
        a2 (float) - upper value 

    Returns:
        time (float) - mean switching time.
    """
    values = ((arr >= a2).astype(int) - (arr <= a1).astype(int))
    sign = np.sign(values)
    sz = (sign == 0)
    while sz.any():
        sign[sz] = np.roll(sign, 1)[sz]
        sz = (sign == 0)

    signchange = ((np.roll(sign, 1) - sign) != 0).astype(int)
    signchange[0] = 0
    return len(arr) / signchange.sum()


def switching_time_dist(arr, a1, a2):
    """
    Returns the switching time distribution for an array arr between two values.

    Args:
        arr (np.array) -
        a1 (float) - lower value
        a2 (float) - upper value 
    
    Returns:
        dist (np.array) - an array of switching times
    """
    values = ((arr >= a2).astype(int) - (arr <= a1).astype(int))  # Find times

    asign = np.sign(values)  # Get sign of values (+1 above a2, -1 below a1)
    sz = (asign == 0)  # Find intermediate values
    while sz.any():
        asign[sz] = np.roll(asign, 1)[sz]  # Intermediate values get replaced with what preceeded it.
        sz = (asign == 0)

    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)  # Look for sign changes

    condition = (asign == 1)
    tops = np.diff(np.where(np.concatenate(([condition[0]],
                                            condition[:-1] != condition[1:],
                                            [True])))[0])[::2]

    condition = (asign == -1)
    bottoms = np.diff(np.where(np.concatenate(([condition[0]],
                                               condition[:-1] != condition[1:],
                                               [True])))[0])[::2]
    return np.concatenate([tops, bottoms])


def _zero_runs(arr):
    """
    Returns the indices of the start and end of a run of zeros of an array.

    Args:
        arr (np.array) - 

    Returns:
        ranges (np.ndarray) - indices pairs to mark the start and end of a run of zeros in arr.
    """

    # Creates an array which is 1 where a=0, and pads the ends with a 0.
    iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def switching_points(arr, a1, a2):
    """Returns indices pairs of the transitional periods of a between two values.

    Args:
        arr (np.array) -
        a1 (float) - lower value
        a2 (float) - upper value 

    Returns:
        swpoints (np.ndarray) - indices pairs to mark the start and end of a switch from a1 to a2.
    """
    values = ((arr >= a2).astype(int) - (arr <= a1).astype(int))
    zeros = _zero_runs(values)

    lower = np.maximum(zeros[:, 0] - 1, 0)
    upper = np.minimum(zeros[:, 1] + 1, len(zeros) - 1)

    swpoints = zeros[np.abs(values[lower] - values[upper]) == 2]
    return swpoints


def angular_velocity(x1, x2, t, sample_size, deviations=True):
    """
    Calculates the angular velocity for a given t.

    This is given by the correlation <x1(s)x2(s+t)> - <x1(s+t)x2(s)>.
    The equivalent can be derived for the deviations from the mean.

    Args:
        x1, x2 (np.array) - Timeseries for x1, x2.
        t (int) - Time difference which correlation is calculated over.
        sample_size - The number of samples used to average over (max = max(len(x1),len(x2)) - sample_size - t).
        deviations (bool) - If True, use the deviations from the mean rather than true values.

    Returns:
        angvec_mean, angvec_std (float, float)
    """

    if deviations:
        x1 = x1 - x1.mean()
        x2 = x2 - x2.mean()

    angvec = x1[:sample_size] * x2[t:sample_size + t] - x1[t:sample_size + t] * x2[:sample_size]
    return angvec.mean(), angvec.std()
