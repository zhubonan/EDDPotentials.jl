using EDDP
training_opts = EDDP.TrainingOptions(nmodels=256, n_nodes=[5], rmse_threshold=0.5)
feature_opts = EDDP.FeatureOptions(
    elements=[:B],
    p2=[2, 4, 6, 8],
    p3=[2, 4, 6, 8],
    q3=[2, 4, 6, 8],
    rcut2=3.75,
    rcut3=3.75,
    )
opts = EDDP.BuildOptions(
seedfile="8B.cell",
per_generation = 100,
shake_per_minima = 10,
build_timeout=1,
shake_amp = 0.02,
shake_cell_amp = 0.02,
workdir = "./",
n_initial=100,
mpinp=2,
n_parallel=1,
relax_extra_opts=Dict{Symbol, Any}(:categories=>["24-core", "4-core"], :priority=>100),
dft_mode="disp-castep"
)

