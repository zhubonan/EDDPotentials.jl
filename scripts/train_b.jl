##
using StatsBase
using EDDP
using Test

path = "examples/new-test/iter*-dft/*.res" 
path = "examples/new-test/all-res/*.res.gz" 
@time sc = EDDP.StructureContainer([path])
feature_opts = EDDP.FeatureOptions(
    elements=[:B],
    p2=[2, 4, 6, 8],
    p3=[2, 4, 6, 8],
    q3=[2, 4, 6, 8],
    rcut2=3.75,
    rcut3=3.75,
    )


cf = EDDP.CellFeature(feature_opts)

fc = EDDP.FeatureContainer(sc, cf)

##

##
ftrain, fttest, fvalidate = split(fc, 0.8, 0.1, 0.1)

EDDP.standardize!(ftrain, fttest, fvalidate)

# Scale X

model = EDDP.ManualFluxBackPropInterface(cf, 5;yt=ftrain.yt)

res = EDDP.train!(model, ftrain, fttest;show_progress=true);