module SparseMatricesCOO

export SparseMatrixCOO, coo_spzeros

# TODO: fix jldoctests
# TODO: implement views

using LinearAlgebra, SparseArrays

# we use UnicodePlots to `show` a sparse matrix instead of displaying its entries
# this should be revisited after https://github.com/JuliaLang/julia/pull/33821
using UnicodePlots

using MKL_jll

function __init__()
  ccall((:MKL_Set_Interface_Layer, libmkl_rt), Cint, (Cint,), Base.USE_BLAS64 ? 1 : 0)
end

include("coo_utils.jl")
include("coo_types.jl")
include("coo_linalg.jl")

if Sys.islinux() || Sys.isapple() || Sys.iswindows()
  include("coo_mkl_wrapper.jl")
  include("coo_mkl_interface.jl")
end

end # module
