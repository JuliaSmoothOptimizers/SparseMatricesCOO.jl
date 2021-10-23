using JSOTemplate
using Documenter

DocMeta.setdocmeta!(JSOTemplate, :DocTestSetup, :(using JSOTemplate); recursive = true)

makedocs(;
  modules = [JSOTemplate],
  doctest = true,
  linkcheck = false,
  strict = false,
  authors = "Abel Soares Siqueira <abel.s.siqueira@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/JSOTemplate.jl/blob/{commit}{path}#{line}",
  sitename = "JSOTemplate.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/JSOTemplate.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/JSOTemplate.jl",
  push_preview = true,
  devbranch = "main",
)
