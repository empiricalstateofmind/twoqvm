from unittest import TestCase
import numpy as np
from twoqvm import *

class TestFiniteModelConstruction(TestCase):
	"""Testing the construction of the 2qVM model."""

	def test_finite_model_with_integer_input(self):
		for N, S, Z in zip([10, 100, 1000], [3, 20, 100], [2, 30, 400]):
			model = TwoQVoterModel(N=N, S=S, Z=Z, q=(1,2), rho=0)
			self.assertTrue(model.s[0] == model.s[1])
			self.assertTrue(model.z[0] == model.z[1])

	def test_finite_model_with_tuple_input(self):
		for N, S, Z in zip([10, 100, 1000], [3, 20, 100], [2, 30, 400]):
			model = TwoQVoterModel(N=N, S=(S,S), Z=(Z,Z), q=(1,2), rho=0)
			self.assertTrue(model.s[0] == model.s[1])
			self.assertTrue(model.z[0] == model.z[1])

	def test_assertions(self):
		with self.assertRaises(AssertionError):
			model = TwoQVoterModel(N=100, S=(99,1), Z=(20,20), q=(1,2), rho=0)
		with self.assertRaises(AssertionError):
			model = TwoQVoterModel(N=100, S=(0,50), Z=(25,25), q=(1,2), rho=0)
		with self.assertRaises(AssertionError):
			model = TwoQVoterModel(N=100, S=(25,25), Z=(25,25), q=(1,2), rho=-1)
		with self.assertRaises(AssertionError):
			model = TwoQVoterModel(N=100, S=(25,25), Z=(25,25), q=(1,2), rho=(0.5, 5))

	def test_simulation_setup(self):
		model = TwoQVoterModel(N=100, S=(25,25), Z=(25,25), q=(1,2), rho=0.5, max_iterations=100)
		self.assertIsInstance(model.x1_track, np.ndarray)
		self.assertIsInstance(model.x2_track, np.ndarray)
		self.assertTrue(len(model.x1_track)==model.max_iterations)
		self.assertTrue(len(model.x2_track)==model.max_iterations)

		model = TwoQVoterModel(N=100, S=(25,25), Z=(25,25), q=(1,2), rho=0.5)
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

	def test_diffision_matrix(self):
		pass

	def test_correlation_matrix(self):
		pass

	def test_angular_momentum(self):
		pass

	def test_run_iterations(self):
		pass

	def test_angular_velocity(self):
		pass

class TestInfiniteMethods(TestCase):
	"""Testing methods associated with the finite model."""

	def test_drift_matrix(self):
		pass

	def test_fixed_point_function(self):
		pass