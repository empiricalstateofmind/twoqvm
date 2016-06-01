# 2qVZ Model

A class to model, simulate, and analyse the 2q Voter Model with Zealots (2qVZ).

When using this software, please cite the following papers:

[**Characterization of the nonequilibrium steady state of a heterogeneous nonlinear q-voter model with zealotry (2016)**](http://dx.doi.org/10.1209/0295-5075/113/48001)
_Andrew Mellor, Mauro Mobilia, R.K.P. Zia_
European Physics Letters

---

## Usage 

The model can be called simply with
``` python
model = TwoQVoterModel(N=100, S=30, Z=20, q=(1,2), rho=0.5)
model.run_iterations(1000)

plt.plot(model.x1_track)
plt.plot(model.x2_track)
```
or alternatively with variable numbers of S1,S2 agents and Zp and Zm zealots
``` python
model = TwoQVoterModel(N=100, S=(25,35), Z=(15,25), q=(1,2), rho=(0.0,1.0))
model.run_iterations(1000)

plt.plot(model.x1_track)
plt.plot(model.x2_track)
```
To consider the infinite system the model can be called with
``` python
model = InfiniteTwoQVoterModel(S=0.3, Z=0.2, q=(1,2))
```
which gives access to methods for finding fixed points/stability but for obvious reasons does not allow for simulation/diffusion.
 
### Available Methods

* Calculate the fixed points of the system
* Calculate the stability matrix F at the fixed points
* Calculate the diffusion matrix D at the fixed points
* Calculate the two-point correlation functions
* Calculate the switching time between the two fixed points (for z<z_c)
* Calculate z_c and the effective q
* Calculate the exact stationary distribution numerically
* Calculate probability currents at stationarity.