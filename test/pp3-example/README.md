## Fitting a Lennard-Jones potential

In this example we train an two-body EDDP for a simple Lennard-Jones potential

Content of the `Al.pp` file:

```
1 12 6 2.5
Al
# Epsilon
1
# Sigma
2
# Beta
1
```

Defines a shifted LJ potential with a cut-off radii of 2.5 \sigma (5 \AA).

The features defined in `link.toml`

```
[cf]
elements = ["Al"]
rcut2 = 5.0
rcut3 = 5.0
p2 = [6, 18, 10]
q3 = []
p3 = []
geometry_sequence = true
```

The Lennard-Jones potentials are very "hard-sphere-like" when the particles are close to each other.
We don't want our potential to focus on minimising the errors in this region.
First, a filters is added with the `energy_threshold` keyword which filters away structures above a certain energy per atom (relative to the lowest energy structure).
Second, a Boltzmann weighting is applied using the `boltzmann_kt` keyword that reduces the weight of the high-energy training structure.
One should note that the interactions in the actual DFT calculations are unlikely to be as hard as the Lennard-Jone potentials here.