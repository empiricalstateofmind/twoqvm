import numpy as np
from scipy.integrate import quad, dblquad


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
        asign[sz] = np.roll(asign, 1)[sz]  # Intermediate values get replaced with what preceded it.
        sz = (asign == 0)

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
        x1 (np.array) - Time series for x1.
        x2 (np.array) - Time series for x2.
        t (int) - Time difference which correlation is calculated over.
        sample_size (int) - The number of samples used to average over (max = max(len(x1),len(x2)) - sample_size - t).
        deviations (bool) - If True, use the deviations from the mean rather than true values.

    Returns:
        angvec_mean, angvec_std (float, float)
    """

    if deviations:
        x1 = x1 - x1.mean()
        x2 = x2 - x2.mean()

    angvec = x1[:sample_size] * x2[t:sample_size + t] - x1[t:sample_size + t] * x2[:sample_size]
    return angvec.mean(), angvec.std()


def mean_switching_time_estimate(N, a, b, z, q, kind):

    if kind.lower() == 'effective':

        def x2(x1, s, q1=1, q2=2):
            return s / (1 + (s / x1 - 1) ** (q1 / q2))

        def Tp(x1, z, q):
            s = (1 - 2 * z) / 2
            return (s - x1) * (z + x1 + x2(x1, s)) ** q

        def Tm(x1, z, q):
            s = (1 - 2 * z) / 2
            return x1 * (1 - z - x1 - x2(x1, s)) ** q
    elif kind.lower() == 'theta':

        def Tp(x, z, q):
            return ((1 - 2 * z) - x) * (x + z) ** q

        def Tm(x, z, q):
            return x * ((1 - 2 * z) - x + z) ** q

    return _full(N, a, b, z, q)


def _phi(v, a, z, q):
    f = lambda u, z, q: (Tm(u, z, q) - Tp(u, z, q)) / (Tm(u, z, q) + Tp(u, z, q))
    return -2 * quad(f, a, v, args=(z, q))[0]


def _inner(v, N, z, q, lb):
    return np.exp(N * _phi(v, lb, z, q)) / (Tp(v, z, q) + Tm(v, z, q))


def _outer(y, N, z, q, lb):
    return np.exp(-N * _phi(y, lb, z, q))


def _full(N, a, b, z, q):
    return 2 * N * \
           dblquad(lambda x, y, z, q: _inner(x, N, z, q, a) * _outer(y, N, z, q, a), a, b, lambda y: a, lambda y: y,
                   args=(z, q))[0]
