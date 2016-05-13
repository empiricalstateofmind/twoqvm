from unittest import TestCase
from twoqvm import *

class TestModelConstruction(TestCase):
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

