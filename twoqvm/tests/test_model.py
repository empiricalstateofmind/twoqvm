from unittest import TestCase

from twoqvm import *


class TestFiniteModelConstruction(TestCase):
    """Testing the construction of the 2qVM model."""

    def test_finite_model_with_integer_input(self):
        for N, S, Z in zip([10, 100, 1000], [3, 20, 100], [2, 30, 400]):
            model = TwoQVoterModel(N=N, S=S, Z=Z, q=(1, 2), rho=0)
            self.assertTrue(model.s[0] == model.s[1])
            self.assertTrue(model.z[0] == model.z[1])

    def test_finite_model_with_tuple_input(self):
        for N, S, Z in zip([10, 100, 1000], [3, 20, 100], [2, 30, 400]):
            model = TwoQVoterModel(N=N, S=(S, S), Z=(Z, Z), q=(1, 2), rho=0)
            self.assertTrue(model.s[0] == model.s[1])
            self.assertTrue(model.z[0] == model.z[1])

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            TwoQVoterModel(N=100, S=(99, 1), Z=(20, 20), q=(1, 2), rho=0)
        with self.assertRaises(AssertionError):
            TwoQVoterModel(N=100, S=(0, 50), Z=(25, 25), q=(1, 2), rho=0)
        with self.assertRaises(AssertionError):
            TwoQVoterModel(N=100, S=(25, 25), Z=(25, 25), q=(1, 2), rho=-1)
        with self.assertRaises(AssertionError):
            TwoQVoterModel(N=100, S=(25, 25), Z=(25, 25), q=(1, 2), rho=(0.5, 5))

    def test_simulation_setup(self):
        model = TwoQVoterModel(N=100, S=(25, 25), Z=(25, 25), q=(1, 2), rho=0.5, max_iterations=100)
        self.assertIsInstance(model.x1_track, np.ndarray)
        self.assertIsInstance(model.x2_track, np.ndarray)
        self.assertTrue(len(model.x1_track) == model.max_iterations + 1)
        self.assertTrue(len(model.x2_track) == model.max_iterations + 1)

        model = TwoQVoterModel(N=100, S=(25, 25), Z=(25, 25), q=(1, 2), rho=0.5)
        self.assertIsInstance(model.x1_track, list)
        self.assertIsInstance(model.x2_track, list)


class TestInfiniteModelConstruction(TestCase):
    """Testing the construction of the infinite 2qVM model."""

    def test_infinite_model_with_float_input(self):
        for S, Z in zip([0.4, 0.3, 0.2], [0.1, 0.2, 0.3]):
            model = InfiniteTwoQVoterModel(s=S, z=Z, q=(1, 2))
            self.assertTrue(model.s[0] == model.s[1])
            self.assertTrue(model.z[0] == model.z[1])

    def test_infinite_model_with_tuple_input(self):
        for S, Z in zip([0.4, 0.3, 0.2], [0.1, 0.2, 0.3]):
            model = InfiniteTwoQVoterModel(s=(S, S), z=(Z, Z), q=(1, 2))
            self.assertTrue(model.s[0] == model.s[1])
            self.assertTrue(model.z[0] == model.z[1])

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            model = InfiniteTwoQVoterModel(s=(0.5, 0.4), z=(0.1, 0.2), q=(1, 2))


class TestFiniteMethods(TestCase):
    """Testing methods associated with the finite model."""

    def setUp(self):
        model1 = TwoQVoterModel(N=100, S=40, Z=10, q=(1, 2), rho=0.1)
        model2 = TwoQVoterModel(N=100, S=30, Z=20, q=(1, 2), rho=0.1)
        model3 = TwoQVoterModel(N=100, S=(40, 20), Z=(15, 25), q=(1, 2), rho=0.1)
        self.models = [model1, model2, model3]

    def tearDown(self):
        self.models = None

    def test_diffision_matrix(self):
        pass

    def test_correlation_matrix(self):
        pass

    def test_angular_momentum(self):
        pass

    def test_run_iterations(self):
        for model in self.models:
            model.run_iterations(500)
            assert all([x >= 0 for x in model.x1_track])
            assert all([x <= model.s[0] for x in model.x1_track])
            assert all([x >= 0 for x in model.x2_track])
            assert all([x <= model.s[1] for x in model.x2_track])

    def test_angular_velocity(self):
        pass


class TestInfiniteMethods(TestCase):
    """Testing methods associated with the finite model."""

    def test_q_eff(self):
        pass

    def test_z_c(self):
        pass

    def test_drift_matrix(self):
        pass

    def test_fixed_point_function(self):
        pass
