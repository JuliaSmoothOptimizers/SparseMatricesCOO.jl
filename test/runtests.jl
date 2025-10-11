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

@testset "Conversion from another SparseMatrixCOO" begin
  T = Float64
  A = sprand(T, 10, 15, 0.4)
  B = SparseMatrixCOO(A)
  B32_1 = convert(SparseMatrixCOO{Float32, Int}, B)
  @test typeof(B32_1) == SparseMatrixCOO{Float32, Int}
  @test typeof(B32_1.vals) == Vector{Float32}
  B32_2 = convert(SparseMatrixCOO{Float32, Int32}, B)
  @test typeof(B32_2) == SparseMatrixCOO{Float32, Int32}
  @test typeof(B32_2.rows) == Vector{Int32}
  @test typeof(B32_2.cols) == Vector{Int32}
  @test typeof(B32_2.vals) == Vector{Float32}
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
  allocs = @allocated mul!(yB, B, x)
  @test all(yA .≈ yB)
  @test allocs == 0

  y = rand(m)
  xA = Vector{T}(undef, n)
  mul!(xA, transpose(A), y)
  xB = Vector{T}(undef, n)
  mul!(xB, transpose(B), y)
  @test all(xA .≈ xB)
  mul!(xA, A', y)
  mul!(xB, B', y)
  @test all(xA .≈ xB)

  yA = Array{T}(undef, m, n)
  x = rand(n, n)
  mul!(yA, A, x)
  yB = Array{T}(undef, m, n)
  mul!(yB, B, x)
  allocs = @allocated mul!(yB, B, x)
  @test all(yA .≈ yB)
  @test allocs == 0
  xA = Array{T}(undef, n, m)
  y = rand(m, m)
  mul!(xA, transpose(A), y)
  xB = Array{T}(undef, n, m)
  mul!(xB, transpose(B), y)
  @test all(xA .≈ xB)
  allocs = @allocated mul!(yB, B, x)
  @test allocs == 0

  A = sprand(ComplexF64, m, n, 0.4)
  B = SparseMatrixCOO(A)
  T = eltype(A)
  @test eltype(B) == T
  y = rand(m)
  xA = Vector{T}(undef, n)
  mul!(xA, transpose(A), y)
  xB = Vector{T}(undef, n)
  mul!(xB, transpose(B), y)
  @test all(xA .≈ xB)
  mul!(xA, A', y)
  mul!(xB, B', y)
  allocs = @allocated mul!(xB, B', y)
  @test all(xA .≈ xB)
  @test allocs == 0
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
  yB = ones(m)
  allocs = @allocated mul!(yB, B, x, α, β)
  @test all(yA .≈ yB)
  @test allocs == 0
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
  yB = ones(m)
  allocs = @allocated mul!(yB, lB, x, α, β)
  @test all(yA .≈ yB)
  @test allocs == 0

  T = ComplexF64
  A = sprand(T, m, m, 0.4)
  A[diagind(A)] .= real.(A[diagind(A)])
  B = SparseMatrixCOO(A)
  lA = Hermitian(A, :L)
  lB = Hermitian(B, :L)
  x = rand(T, m)
  yA = ones(T, m)
  α = rand()
  β = rand()
  mul!(yA, lA, x, α, β)
  yB = ones(T, m)
  mul!(yB, lB, x, α, β)
  yB = ones(T, m)
  allocs = @allocated mul!(yB, lB, x, α, β)
  @test all(yA .≈ yB)
  @test allocs == 0

  n = 7
  x = rand(T, m, n)
  yA = ones(T, m, n)
  mul!(yA, lA, x, α, β)
  yB = ones(T, m, n)
  mul!(yB, lB, x, α, β)
  yB = ones(T, m, n)
  allocs = @allocated mul!(yB, lB, x, α, β)
  @test all(yA .≈ yB)
  @test allocs == 0
end

@testset "special multiply tests" begin
  A = sprand(10, 15, 0.4)
  D1 = Diagonal(rand(10))
  A_coo = SparseMatrixCOO(A)
  lmul!(D1, A)
  lmul!(D1, A_coo)
  @test norm(A - A_coo) ≤ sqrt(eps()) * norm(A)
  ldiv!(D1, A)
  ldiv!(D1, A_coo)
  @test norm(A - A_coo) ≤ sqrt(eps()) * norm(A)

  D2 = Diagonal(rand(15))
  rmul!(A, D2)
  rmul!(A_coo, D2)
  @test norm(A - A_coo) ≤ sqrt(eps()) * norm(A)
  rdiv!(A, D2)
  rdiv!(A_coo, D2)
  @test norm(A - A_coo) ≤ sqrt(eps()) * norm(A)
end

@testset "dropzeros" begin
  rows = [1, 1, 3, 2, 5, 7, 9, 6]
  cols = [1, 2, 2, 3, 7, 9, 9, 12]
  vals = [0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0]
  A = SparseMatrixCOO(10, 15, rows, cols, vals)
  dropzeros!(A)
  @test length(A.rows) == 3
  @test A.rows == [1, 3, 9]
  @test A.cols == [2, 2, 9]
  @test A.vals == [2.0, 3.0, 2.0]
end

@testset "cat" begin
  A = sprand(Float64, 50, 40, 0.4)
  B = sprand(Float64, 50, 20, 0.4)
  C = sprand(Float64, 30, 40, 0.4)
  csc_hcat = [A B A]
  coo_hcat = [SparseMatrixCOO(A) SparseMatrixCOO(B) SparseMatrixCOO(A)]
  csc_vcat = [A; C; A]
  coo_vcat = [SparseMatrixCOO(A); SparseMatrixCOO(C); SparseMatrixCOO(A)]
  @test norm(csc_hcat - coo_hcat) ≤ sqrt(eps()) * norm(csc_hcat)
  @test norm(csc_vcat - coo_vcat) ≤ sqrt(eps()) * norm(csc_vcat)
  @test issorted(coo_vcat.cols)

  csc_hcat = [A 3.0 * I]
  coo_hcat = [SparseMatrixCOO(A) 3.0 * I]
  csc_vcat = [A; 3.0 * I]
  coo_vcat = [SparseMatrixCOO(A); 3.0 * I]
  @test norm(csc_hcat - coo_hcat) ≤ sqrt(eps()) * norm(csc_hcat)
  @test norm(csc_vcat - coo_vcat) ≤ sqrt(eps()) * norm(csc_vcat)
  @test issorted(coo_vcat.cols)
  csc_hcat = [I A]
  coo_hcat = [I SparseMatrixCOO(A)]
  csc_vcat = [I; A]
  coo_vcat = [I; SparseMatrixCOO(A)]
  @test norm(csc_hcat - coo_hcat) ≤ sqrt(eps()) * norm(csc_hcat)
  @test norm(csc_vcat - coo_vcat) ≤ sqrt(eps()) * norm(csc_vcat)
  @test issorted(coo_vcat.cols)

  A = sprand(Float64, 50, 40, 0.4)
  B = sprand(Float64, 50, 20, 0.4)
  D = sprand(Float64, 30, 20, 0.4)
  csc_cat = [A B; spzeros(Float64, 30, 40) D]
  coo_cat =
    vcat([SparseMatrixCOO(A) SparseMatrixCOO(B)], [coo_spzeros(Float64, 30, 40) SparseMatrixCOO(D)])
  @test norm(csc_cat - coo_cat) ≤ sqrt(eps()) * norm(csc_cat)
  @test issorted(coo_cat.cols)
end

@testset "arithmetic operations" begin
  A = sprand(Float64, 20, 20, 0.1)

  # test -
  A_coo = SparseMatrixCOO(A)
  @test norm(-A - (-A_coo)) ≤ sqrt(eps()) * norm(A)

  # test + with diagonal matrix
  D = Diagonal(rand(20))
  B_csc = D + A
  B_coo = D + A_coo
  @test norm(B_csc - B_coo) ≤ sqrt(eps()) * norm(B_csc)
  @test issorted(B_coo.cols)

  B = sprand(Float64, 20, 20, 0.1)
  C_csc = A + B
  C_coo = A_coo + SparseMatrixCOO(B)
  @test norm(C_csc - C_coo) ≤ sqrt(eps()) * norm(C_csc)
end

@testset "row/col reduce" begin
  A = sprand(Float64, 10, 15, 0.2) .- 0.5
  A_coo = SparseMatrixCOO(A)
  v = zeros(10)
  v_coo = zeros(10)
  maximum!(abs, v, A)
  maximum!(abs, v_coo, A_coo)
  @test norm(v - v_coo) ≤ sqrt(eps()) * norm(v)

  v2 = zeros(15)
  v_coo2 = zeros(15)
  maximum!(abs, v2', A)
  maximum!(abs, v_coo2', A_coo)
  @test norm(v2 - v_coo2) ≤ sqrt(eps()) * norm(v2)

  As = Symmetric(A * A')
  As_coo = Symmetric(SparseMatrixCOO(As.data))
  v .= 0
  v_coo .= 0
  maximum!(abs, v, As)
  maximum!(abs, v_coo, As_coo)
  @test norm(v - v_coo) ≤ sqrt(eps()) * norm(v)
  v .= 0
  v_coo .= 0
  maximum!(abs, v', As)
  maximum!(abs, v_coo', As_coo)
  @test norm(v - v_coo) ≤ sqrt(eps()) * norm(v)

  As = Symmetric(tril(A * A'), :L)
  As_coo = Symmetric(SparseMatrixCOO(As.data), :L)
  v .= 0
  v_coo .= 0
  maximum!(abs, v, As)
  maximum!(abs, v_coo, As_coo)
  @test norm(v - v_coo) ≤ sqrt(eps()) * norm(v)
  v .= 0
  v_coo .= 0
  maximum!(abs, v', As)
  maximum!(abs, v_coo', As_coo)
  @test norm(v - v_coo) ≤ sqrt(eps()) * norm(v)
end

@testset "Kronecker product" begin
  A = sprand(Float64, 10, 15, 0.2)
  B = sprand(Float64, 5, 7, 0.3)
  A_coo = SparseMatrixCOO(A)
  B_coo = SparseMatrixCOO(B)
  C = kron(A, B)
  C_coo = kron(A_coo, B_coo)
  @test norm(C - C_coo) ≤ sqrt(eps()) * norm(C)
end
