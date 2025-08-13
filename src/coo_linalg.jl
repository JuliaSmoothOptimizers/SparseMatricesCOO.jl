function coo_mul!(C::AbstractVector, Arows, Acols, Avals, B::AbstractVector, α, Annz)
  @inbounds for k = 1:Annz
    i, j = Arows[k], Acols[k]
    C[i] += α * Avals[k] * B[j]
  end
end

function coo_mul!(C::AbstractMatrix, Arows, Acols, Avals, B::AbstractMatrix, α, Annz)
  @inbounds for k = 1:Annz
    i, j = Arows[k], Acols[k]
    @views C[i, :] .+= (α * Avals[k]) .* B[j, :]
  end
end

for T in (
  AbstractVector,
  AbstractMatrix,
  Union{Bidiagonal, SymTridiagonal, Tridiagonal},
  Diagonal,
  Transpose{<:Any, <:AbstractVecOrMat},
  Adjoint{<:Any, <:AbstractVecOrMat},
)
  @eval function LinearAlgebra.mul!(
    C::StridedVecOrMat,
    A::AbstractSparseMatrixCOO,
    B::$T,
    α::Number,
    β::Number,
  )
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    if β != 1
      β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    coo_mul!(C, A.rows, A.cols, A.vals, B, α, nnz(A))
    C
  end
end

function coo_adjtrans_mul!(C::AbstractVector, Arows, Acols, Avals, B::AbstractVector, α, Annz, t)
  @inbounds for k = 1:Annz
    i, j = Acols[k], Arows[k]
    C[i] += α * t(Avals[k]) * B[j]
  end
end

function coo_adjtrans_mul!(C::AbstractMatrix, Arows, Acols, Avals, B::AbstractMatrix, α, Annz, t)
  @inbounds for k = 1:Annz
    i, j = Acols[k], Arows[k]
    @views C[i, :] .+= (α * t(Avals[k])) .* B[j, :]
  end
end

for (T, t) in ((Adjoint, adjoint), (Transpose, transpose))
  for Tb in (
    AbstractVector,
    AbstractMatrix,
    Union{Bidiagonal, SymTridiagonal, Tridiagonal},
    Diagonal,
    Transpose{<:Any, <:AbstractVecOrMat},
    Adjoint{<:Any, <:AbstractVecOrMat},
  )
    @eval function LinearAlgebra.mul!(
      C::StridedVecOrMat,
      xA::$T{<:Any, <:AbstractSparseMatrixCOO},
      B::$Tb,
      α::Number,
      β::Number,
    )
      A = xA.parent
      size(A, 2) == size(C, 1) || throw(DimensionMismatch())
      size(A, 1) == size(B, 1) || throw(DimensionMismatch())
      size(B, 2) == size(C, 2) || throw(DimensionMismatch())
      if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
      end
      coo_adjtrans_mul!(C, A.rows, A.cols, A.vals, B, α, nnz(A), $t)
      C
    end
  end
end

function coo_sym_mul!(C::AbstractVector, Arows, Acols, Avals, B::AbstractVector, α, Annz, t, uplo)
  @inbounds for k = 1:Annz
    i, j, a = Arows[k], Acols[k], Avals[k]
    ((uplo == 'U' && i > j) || (uplo == 'L' && i < j)) && continue  # ignore elements in this triangle
    C[i] += α * a * B[j]
    if i != j
      C[j] += α * t(a) * B[i]
    end
  end
end

function coo_sym_mul!(C::AbstractMatrix, Arows, Acols, Avals, B::AbstractMatrix, α, Annz, t, uplo)
  @inbounds for k = 1:Annz
    i, j, a = Arows[k], Acols[k], Avals[k]
    ((uplo == 'U' && i > j) || (uplo == 'L' && i < j)) && continue  # ignore elements in this triangle
    @views C[i, :] .+= (α * a) .* B[j, :]
    if i != j
      @views C[j, :] .+= (α * t(a)) .* B[i, :]
    end
  end
end

for (T, t) in ((Hermitian, adjoint), (Symmetric, transpose))
  for Tb in (
    AbstractVector,
    AbstractMatrix,
    Union{Bidiagonal, SymTridiagonal, Tridiagonal},
    Diagonal,
    Transpose{<:Any, <:AbstractVecOrMat},
    Adjoint{<:Any, <:AbstractVecOrMat},
  )
    @eval function LinearAlgebra.mul!(
      C::StridedVecOrMat,
      xA::$T{<:Any, <:AbstractSparseMatrixCOO},
      B::$Tb,
      α::Number,
      β::Number,
    )
      A = xA.data
      size(A, 2) == size(B, 1) || throw(DimensionMismatch())
      size(A, 1) == size(C, 1) || throw(DimensionMismatch())
      size(B, 2) == size(C, 2) || throw(DimensionMismatch())
      if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
      end
      coo_sym_mul!(C, A.rows, A.cols, A.vals, B, α, nnz(A), $t, xA.uplo)
      C
    end
  end
end

function loperation!(operation::Function, D::Diagonal{T}, A::SparseMatrixCOO{T}) where {T}
  @assert size(D, 2) == size(A, 1)
  nnz_A = nnz(A)
  Arows, Avals = A.rows, A.vals
  d = D.diag
  for k = 1:nnz_A
    i = Arows[k]
    Avals[k] = operation(Avals[k], d[i])
  end
  return A
end
LinearAlgebra.lmul!(D::Diagonal, A::SparseMatrixCOO) = loperation!(*, D, A)
LinearAlgebra.ldiv!(D::Diagonal, A::SparseMatrixCOO) = loperation!(/, D, A)

function roperation!(operation::Function, A::SparseMatrixCOO{T}, D::Diagonal{T}) where {T}
  @assert size(D, 1) == size(A, 2)
  nnz_A = nnz(A)
  Acols, Avals = A.cols, A.vals
  d = D.diag
  for k = 1:nnz_A
    j = Acols[k]
    Avals[k] = operation(Avals[k], d[j])
  end
  return A
end
LinearAlgebra.rmul!(A::SparseMatrixCOO, D::Diagonal) = roperation!(*, A, D)
LinearAlgebra.rdiv!(A::SparseMatrixCOO, D::Diagonal) = roperation!(/, A, D)

function SparseArrays.dropzeros!(A::SparseMatrixCOO{T}) where {T}
  Arows, Acols, Avals = A.rows, A.cols, A.vals
  Awritepos = 0
  nnzA = length(Arows)
  @inbounds for k = 1:nnzA
    Ax = Avals[k]
    if Ax != zero(T)
      Awritepos += 1
      Arows[Awritepos] = Arows[k]
      Acols[Awritepos] = Acols[k]
      Avals[Awritepos] = Ax
    end
  end
  if Awritepos != nnzA
    resize!(Arows, Awritepos)
    resize!(Acols, Awritepos)
    resize!(Avals, Awritepos)
  end
end

import Base.hcat, Base.vcat

function hcat(A::SparseMatrixCOO{T}, B::SparseMatrixCOO{T}) where {T}
  mA, nA = size(A)
  mB, nB = size(B)
  @assert mA == mB
  rows = [A.rows; B.rows]
  cols = [A.cols; B.cols .+ nA]
  vals = [A.vals; B.vals]
  return SparseMatrixCOO(mA, nA + nB, rows, cols, vals)
end

function uniform_scaling_to_coo(λI::UniformScaling, n::Int, T::DataType)
  λ = (λI.λ == true) ? one(T) : λI.λ
  return SparseMatrixCOO(n, n, Vector(1:n), Vector(1:n), fill(T(λ), n))
end

function hcat(A::SparseMatrixCOO{T}, λI::UniformScaling) where {T}
  m, n = size(A)
  return SparseMatrixCOO(
    m,
    n + m,
    [A.rows; 1:m],
    [A.cols; (n + 1):(m + n)],
    [A.vals; fill(λI.λ, m)],
  )
end

function hcat(λI::UniformScaling, A::SparseMatrixCOO{T}) where {T}
  m, n = size(A)
  return SparseMatrixCOO(m, n + m, [1:m; A.rows], [1:m; A.cols .+ m], [fill(λI.λ, m); A.vals])
end

function hcat(As::AbstractSparseMatrixCOO...)
  A = As[1]
  for i = 2:length(As)
    A = [A As[i]]
  end
  return A
end

function vcat(A::SparseMatrixCOO{T, I}, B::SparseMatrixCOO{T, I}) where {T, I}
  mA, nA = size(A)
  mB, nB = size(B)
  nnzA = nnz(A)
  nnzB = nnz(B)
  nnz_tot = nnz(A) + nnz(B)
  Arows, Acols, Avals = A.rows, A.cols, A.vals
  Brows, Bcols, Bvals = B.rows, B.cols, B.vals
  @assert nA == nB
  rows = Vector{I}(undef, nnz_tot)
  cols = Vector{I}(undef, nnz_tot)
  vals = Vector{T}(undef, nnz_tot)
  # keep column-sorted order then each column is row-sorted
  kA, kB = 1, 1
  for k = 1:nnz_tot
    if kA > nnzA
      rows[k], cols[k], vals[k] = Brows[kB] + mA, Bcols[kB], Bvals[kB]
      kB += 1
    elseif kB > nnzB
      rows[k], cols[k], vals[k] = Arows[kA], Acols[kA], Avals[kA]
      kA += 1
    elseif Acols[kA] > Bcols[kB]
      rows[k], cols[k], vals[k] = Brows[kB] + mA, Bcols[kB], Bvals[kB]
      kB += 1
    elseif Acols[kA] ≤ Bcols[kB]
      rows[k], cols[k], vals[k] = Arows[kA], Acols[kA], Avals[kA]
      kA += 1
    end
  end
  return SparseMatrixCOO(mA + mB, nA, rows, cols, vals)
end

vcat(A::SparseMatrixCOO{T}, λI::UniformScaling) where {T} =
  vcat(A, uniform_scaling_to_coo(λI, size(A, 2), T))
vcat(λI::UniformScaling, A::SparseMatrixCOO{T}) where {T} =
  vcat(uniform_scaling_to_coo(λI, size(A, 2), T), A)

function vcat(As::AbstractSparseMatrixCOO...)
  A = As[1]
  for i = 2:length(As)
    A = [A; As[i]]
  end
  return A
end

"""
    coo_spzeros(T, m, n)

Creates a zero `SparseMatrixCOO` of type `T` with `m` rows and `n` columns.
"""
coo_spzeros(T, m, n) = SparseMatrixCOO(m, n, Int[], Int[], T[])

import Base.-, Base.+

-(A::SparseMatrixCOO) = SparseMatrixCOO(A.m, A.n, copy(A.rows), copy(A.cols), .-copy(A.vals))

function +(A::SparseMatrixCOO{T}, D::Diagonal{T, Vector{T}}) where {T}
  nnz_A = nnz(A)
  m, n = size(A)
  @assert m == n == size(D, 2)
  nnz_B = nnz_A + m
  Arows, Acols, Avals = A.rows, A.cols, A.vals
  for k = 1:nnz_A
    if Arows[k] == Acols[k]
      nnz_B -= 1
    end
  end
  Brows = Vector{Int}(undef, nnz_B)
  Bcols = Vector{Int}(undef, nnz_B)
  Bvals = Vector{T}(undef, nnz_B)
  kB = 1 # current B index
  d = D.diag
  kd = 1 # current d index
  for k = 1:nnz_A
    while kd < Acols[k] || (kd == Acols[k] && kd < Arows[k])
      Brows[kB], Bcols[kB], Bvals[kB] = kd, kd, d[kd]
      kB += 1
      kd += 1
    end
    Arowk, Acolk = Arows[k], Acols[k]
    Brows[kB], Bcols[kB] = Arowk, Acolk
    if kd == Arowk == Acolk
      Bvals[kB] = Avals[k] + d[kd]
      kd += 1
    else
      Bvals[kB] = Avals[k]
    end
    kB += 1
  end
  while kd ≤ n
    Brows[kB], Bcols[kB], Bvals[kB] = kd, kd, d[kd]
    kB += 1
    kd += 1
  end
  return SparseMatrixCOO(n, n, Brows, Bcols, Bvals)
end

+(D::Diagonal, A::SparseMatrixCOO) = A + D

function Base.:+(A::SparseMatrixCOO{T1}, B::SparseMatrixCOO{T2}) where {T1<:Number, T2<:Number}
  A.n == B.n || throw(DimensionMismatch())
  A.m == B.m || throw(DimensionMismatch())
  
  T = promote_type(T1, T2)
  
  rowval_colvalA = collect(zip(A.rows, A.cols))
  rowval_colvalB = collect(zip(B.rows, B.cols))

  rowval_colval = union(rowval_colvalA, rowval_colvalB)
  rowval = first.(rowval_colval)
  colval = last.(rowval_colval)

  nzval = similar(rowval, T)
  @inbounds for i in eachindex(rowval)
      nzval[i] = zero(T)
      for j in eachindex(rowval_colvalA)
          if rowval_colvalA[j] == rowval_colval[i]
              nzval[i] += A.vals[j]
              break
          end
      end
      for j in eachindex(rowval_colvalB)
          if rowval_colvalB[j] == rowval_colval[i]
              nzval[i] += B.vals[j]
              break
          end
      end
  end

  return SparseMatrixCOO(A.m, A.n, rowval, colval, nzval)
end

function LinearAlgebra.kron(A::SparseMatrixCOO, B::SparseMatrixCOO)
  mA, nA = size(A)
  mB, nB = size(B)
  out_shape = (mA * mB, nA * nB)
  Annz = nnz(A)
  Bnnz = nnz(B)

  if Annz == 0 || Bnnz == 0
      return SparseMatrixCOO(Int[], Int[], T[], out_shape...)
  end

  row = (A.rows .- 1) .* mB
  row = repeat(row, inner = Bnnz)
  col = (A.cols .- 1) .* nB
  col = repeat(col, inner = Bnnz)
  data = repeat(A.vals, inner = Bnnz)

  row .+= repeat(B.rows, outer = Annz)
  col .+= repeat(B.cols, outer = Annz)

  data .*= repeat(B.vals, outer = Annz)

  return SparseMatrixCOO(out_shape[1], out_shape[2], row, col, data)
end

# maximum! functions
replace_if_minusinf(val::T, replacement::T) where {T} = (val == -T(Inf)) ? replacement : val
function LinearAlgebra.maximum!(f::Function, v::AbstractVector{T}, A::SparseMatrixCOO{T}) where {T}
  nnz_A = nnz(A)
  Arows, Avals = A.rows, A.vals
  @assert length(v) == size(A, 1)
  v .= -T(Inf)
  for k = 1:nnz_A
    i, faij = Arows[k], f(Avals[k])
    max(faij, v[i]) == faij && (v[i] = faij)
  end
  # if empty rows
  replacement = f(zero(T))
  v .= replace_if_minusinf.(v, replacement)
end

function LinearAlgebra.maximum!(
  f::Function,
  v::Union{Transpose{T, <:AbstractVector{T}}, Adjoint{T, <:AbstractVector{T}}},
  A::SparseMatrixCOO{T},
) where {T}
  nnz_A = nnz(A)
  Acols, Avals = A.cols, A.vals
  @assert size(v, 2) == size(A, 2)
  vT = v.parent
  vT .= -T(Inf)
  for k = 1:nnz_A
    j, faij = Acols[k], f(Avals[k])
    max(faij, vT[j]) == faij && (vT[j] = faij)
  end
  # if empty rows
  replacement = f(zero(T))
  v .= replace_if_minusinf.(v, replacement)
end

function maximum_sym_vec!(f::Function, v::AbstractVector{T}, A::SparseMatrixCOO{T}) where {T}
  nnz_A = nnz(A)
  Arows, Acols, Avals = A.rows, A.cols, A.vals
  @assert length(v) == size(A, 1)
  v .= -T(Inf)
  for k = 1:nnz_A
    i, j, faij = Arows[k], Acols[k], f(Avals[k])
    max(faij, v[i]) == faij && (v[i] = faij)
    max(faij, v[j]) == faij && (v[j] = faij)
  end
  # if empty rows
  replacement = f(zero(T))
  v .= replace_if_minusinf.(v, replacement)
end

LinearAlgebra.maximum!(
  f::Function,
  v::AbstractVector{T},
  A::Symmetric{T, <:SparseMatrixCOO{T}},
) where {T} = maximum_sym_vec!(f, v, A.data)

LinearAlgebra.maximum!(
  f::Function,
  v::Union{Transpose{T, <:AbstractVector{T}}, Adjoint{T, <:AbstractVector{T}}},
  A::Symmetric{T, <:SparseMatrixCOO{T}},
) where {T} = maximum_sym_vec!(f, v.parent, A.data)
