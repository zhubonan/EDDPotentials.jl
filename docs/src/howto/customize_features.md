# Customize Features

Feature customization in `EDDPotentials.jl` is centered around `CellFeature`.

## Common adjustments

- change the two-body exponents with `p2`
- change the three-body exponents with `p3` and `q3`
- tune the cutoff radii with `rcut2` and `rcut3`
- disable three-body terms by passing empty `p3` and `q3`

```julia
using EDDPotentials

cf = CellFeature(
    [:C, :H];
    p2=2:6,
    p3=2:4,
    q3=1:3,
    rcut2=4.5,
    rcut3=4.0,
)
```

For a longer walkthrough, see [tutorial_features.md](../tutorial_features.md) and
[Understanding Features](../concepts/features.md).
