using Documenter
using CellTools

makedocs(
    sitename = "CellTools",
    format = Documenter.HTML(),
    modules = [CellTools]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
