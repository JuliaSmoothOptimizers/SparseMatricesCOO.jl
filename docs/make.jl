using SparseMatricesCOO
using Documenter

DocMeta.setdocmeta!(SparseMatricesCOO, :DocTestSetup, :(using SparseMatricesCOO); recursive = true)

makedocs(;
  modules = [SparseMatricesCOO],
  doctest = true,
  linkcheck = false,
  authors = "Dominique Orban <dominique.orban@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/SparseMatricesCOO.jl/blob/{commit}{path}#{line}",
  sitename = "SparseMatricesCOO.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/SparseMatricesCOO.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/SparseMatricesCOO.jl",
  push_preview = true,
  devbranch = "main",
)
