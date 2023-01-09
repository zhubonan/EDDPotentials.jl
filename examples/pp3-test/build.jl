using Revise
using EDDP
using CellBase
using Glob


training_opts = EDDP.TrainingOptions(nmodels=256, n_nodes=[5], rmse_threshold=0.5)
feature_opts = EDDP.FeatureOptions(
    elements=[:B],
    p2=[2, 4, 6, 8, 12],
    p3=[2, 4, 6, 8],
    q3=[2, 4, 6, 8],
    rcut2=3.75,
    rcut3=3.75,
)
opts = EDDP.BuildOptions(
    seedfile="8B.cell",
    per_generation=20,
    shake_per_minima=20,
    build_timeout=1,
    shake_amp=0.02,
    shake_cell_amp=0.02,
    workdir="./",
    n_initial=100,
    mpinp=2,
    n_parallel=1,
    dft_mode="pp3",
)

featurespec = EDDP.CellFeature(feature_opts)
@show opts

EDDP.iterative_build(opts, feature_opts, training_opts)
