using Documenter
using EDDPotentials

makedocs(
    sitename="EDDPotentials.jl",
    format=Documenter.HTML(),
    #modules=[EDDPotentials],
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "AIRSS Background" => "airss.md",
 #      "Functions" => "functions.md",
        "FAQ" => "faq.md",
        "API" => "api.md",
        "Python Tools" => "python_tools.md",
    ],
    warnonly = Documenter.except(:missing_docs, :autodocs_block, :cross_references)
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
if get(ENV, "CI", "") == "true"
    deploydocs(repo="github.com/zhubonan/EDDPotentials.jl")
end
