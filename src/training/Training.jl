module Training

using ..EDDPotentials: to
using ..Potential
using ..Potential.NNLS: nnls
import ..Potential:
    AbstractCellFeatureSpec,
    AbstractNNInterface,
    EDDPCalculator,
    EnsembleNNInterface,
    ManualFluxBackPropInterface,
    predict,
    save_model_artifact,
    setup_fj,
    setup_fj_batched,
    setup_fj_batched_threaded
using ..Potential: ensure_dir
using Base.Threads
using Dates
using JLD2
using Glob
using Printf
using NLSolversBase
using ProgressMeter: Progress
using Parameters
using StatsBase: corspearman, mean, std
using Flux: Dense, Chain
import Flux
import Base
import CellBase
using CellBase: natoms
import Zygote
using Logging
using NonNegLeastSquares
using TimerOutputs

include("data.jl")
include("context.jl")
include("lmsolve.jl")
include("lmsolve_batched.jl")
include("sgd_training.jl")
include("training.jl")

export LevenbergMarquardt,
    BatchedLevenbergMarquardt,
    SGDState,
    AdamState,
    TrainingContext,
    TrainingResults,
    TrainedModel,
    create_ensemble,
    generate_chain,
    generate_f_g_optim,
    train!,
    train_lm!,
    train_lm_batched!,
    train_model,
    train_sgd!,
    progress_callback!,
    nnls_weights,
    predict_energy,
    rmse_per_atom,
    mae_per_atom,
    max_ae_per_atom,
    save_model_artifact

end # module Training
