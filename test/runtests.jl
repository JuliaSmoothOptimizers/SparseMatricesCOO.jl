using LinearAlgebra
using SparseArrays
using Test

using SparseMatricesCOO

include("generic_coo.jl")
if Sys.islinux() || Sys.isapple() || Sys.iswindows()
  include("mkl_coo.jl")
end
