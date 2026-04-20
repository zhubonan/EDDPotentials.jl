using Documenter
using EDDPotentials

makedocs(
    sitename="EDDPotentials.jl",
    format=Documenter.HTML(size_threshold=nothing),
    modules=[EDDPotentials],
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",

        "Concepts" => [
            "What is EDDP?" => "concepts/what_is_eddp.md",
            "Understanding Features" => "concepts/features.md",
            "Iterative Training" => "concepts/iterative_training.md",
        ],

        "Tutorials" => [
            "AIRSS Workflow" => "tutorials/airss_workflow.md",
        ],

        "How-To Guides" => [
            "Customize Features" => "howto/customize_features.md",
            "Crystal Structure Prediction" => "howto/structure_prediction.md",
            "Run Phonon Calculations" => "howto/run_phonon_calculations.md",
            "Enable GPU Acceleration" => "howto/enable_gpu_acceleration.md",
            "Batched LM Training" => "howto/batched_lm_training.md",
        ],

        "Functions" => "functions.md",
    ],
    warnonly = Documenter.except(),
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
if get(ENV, "CI", "") == "true"
    deploydocs(repo="github.com/zhubonan/EDDPotentials.jl")
end
