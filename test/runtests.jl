using LinearAlgebra
using SparseArrays
using Test

using SparseMatricesCOO

@testset "Basic sanity checks" begin
  rs = [1, 2, 3, 4, 4]
  cs = [1, 2, 3, 4, 5]
  vs = Float64.(collect(1:5))
  A = SparseMatrixCOO(4, 5, rs, cs, vs)
  @test eltype(A) == Float64
  @test size(A) == (4, 5)
  @test all(rows(A) .== rs)
  @test all(columns(A) .== cs)
  @test all(values(A) .== vs)
  @test all(nonzeros(A) .== vs)
  @test nnz(A) == length(vs)
  _rs, _cs, _vs = findnz(A)
  @test all(_rs .== rs)
  @test all(_cs .== cs)
  @test all(_vs .== vs)
  fullA = [
    1.0 0.0 0.0 0.0 0.0
    0.0 2.0 0.0 0.0 0.0
    0.0 0.0 3.0 0.0 0.0
    0.0 0.0 0.0 4.0 5.0
  ]
  @test Matrix(A) == fullA
  @test eltype(transpose(A)) == eltype(A)
  _rs, _cs, _vs = findnz(transpose(A))
  @test all(_cs .== rs)
  @test all(_rs .== cs)
  @test all(_vs .== vs)
  @test nnz(A) == nnz(transpose(A))
  @test Matrix(transpose(A)) == transpose(fullA)
  @test eltype(A') == eltype(A)
  _rs, _cs, _vs = findnz(A')
  @test all(_cs .== rs)
  @test all(_rs .== cs)
  @test all(_vs .== vs)
  @test nnz(A) == nnz(A')
  @test Matrix(transpose(A)) == fullA'
  @test convert(SparseMatrixCOO, A) === A
end

@testset "Conversion from dense" begin
  A = Matrix(sprand(10, 15, 0.4))
  B = SparseMatrixCOO(A)
  @test eltype(B) == eltype(A)
  @test size(B) == size(A)
  @test Matrix(B) == A
end

@testset "Conversion from SparseMatrixCSC" begin
  A = sprand(10, 15, 0.4)
  B = SparseMatrixCOO(A)
  @test nnz(B) == nnz(A)
  Ars, Acs, Avs = findnz(A)
  Brs, Bcs, Bvs = findnz(B)
  @test all(Ars .== Brs)
  @test all(Acs .== Bcs)
  @test all(Avs .== Bvs)
end

@testset "3-arg Matrix multiply tests" begin
  m, n = 10, 15
  A = sprand(m, n, 0.4)
  B = SparseMatrixCOO(A)
  T = eltype(A)
  @test eltype(B) == T
  x = rand(n)
  yA = Vector{T}(undef, m)
  mul!(yA, A, x)
  yB = Vector{T}(undef, m)
  mul!(yB, B, x)
  @test all(yA .≈ yB)
  y = rand(m)
  xA = Vector{T}(undef, n)
  mul!(xA, transpose(A), y)
  xB = Vector{T}(undef, n)
  mul!(xB, transpose(B), y)
  @test all(xA .≈ xB)
  mul!(xA, A', y)
  mul!(xB, B', y)
  @test all(xA .≈ xB)
end

@testset "5-arg Matrix multiply tests" begin
  m, n = 10, 15
  A = sprand(m, n, 0.4)
  B = SparseMatrixCOO(A)
  x = rand(n)
  yA = ones(m)
  α = rand()
  β = rand()
  mul!(yA, A, x, α, β)
  yB = ones(m)
  mul!(yB, B, x, α, β)
  @test all(yA .≈ yB)
  y = rand(m)
  xA = ones(n)
  mul!(xA, transpose(A), y, α, β)
  xB = ones(n)
  mul!(xB, transpose(B), y, α, β)
  @test all(xA .≈ xB)
  mul!(xA, A', y, α, β)
  mul!(xB, B', y, α, β)
  @test all(xA .≈ xB)
end

@testset "Symmetric/Hermitian tests" begin
  m = 10
  A = sprand(m, m, 0.4)
  B = SparseMatrixCOO(A)
  lA = Symmetric(A, :L)
  lB = Symmetric(B, :L)
  x = rand(m)
  yA = ones(m)
  α = rand()
  β = rand()
  mul!(yA, lA, x, α, β)
  yB = ones(m)
  mul!(yB, lB, x, α, β)
  @test all(yA .≈ yB)
end
