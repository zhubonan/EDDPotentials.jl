##
using StatsBase
using EDDP

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
tdata = EDDP.training_data(fc;ratio_test=0.1)
tdata.x_train

# Scale X
EDDP.transform_x!(tdata.xt, tdata.x_train)
EDDP.transform_x!(tdata.xt, tdata.x_test)
@test std(reduce(hcat, tdata.x_train)[end, :]) ≈ 1 atol=1e-7
@test mean(reduce(hcat, tdata.x_train)[end, :]) ≈ 0 atol=1e-7

model = EDDP.ManualFluxBackPropInterface(cf, 15, 10, 5;yt=tdata.yt)

res = EDDP.train!(model, tdata.x_train, tdata.y_train;
                  x_test=tdata.x_test, y_test=tdata.y_test, show_progress=true, earlystop=50, maxIter=50)