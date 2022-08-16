module SparseMatricesCOO

export SparseMatrixCOO, coo_spzeros

# TODO: fix jldoctests
# TODO: implement views

using LinearAlgebra, SparseArrays

# we use UnicodePlots to `show` a sparse matrix instead of displaying its entries
# this should be revisited after https://github.com/JuliaLang/julia/pull/33821
using UnicodePlots

include("coo_utils.jl")
include("coo_types.jl")
include("coo_linalg.jl")

end # module
