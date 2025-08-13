module SparseMatricesCOO

export SparseMatrixCOO, coo_spzeros

# TODO: fix jldoctests
# TODO: implement views

using LinearAlgebra, SparseArrays

include("coo_utils.jl")
include("coo_types.jl")
include("coo_linalg.jl")
include("io.jl")

end # module
