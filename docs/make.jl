using Documenter
using EDDP

makedocs(
    sitename="EDDP",
    format=Documenter.HTML(),
    modules=[EDDP],
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Functions" => "functions.md",
        "FAQ" => "faq.md",
        "API" => "api.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
if get("ENV", "CI", "") == "true"
    deploydocs(
        repo = "github.com/zhubonan/EDDP.jl"
    )
end
