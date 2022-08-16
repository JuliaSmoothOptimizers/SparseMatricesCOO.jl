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

function LinearAlgebra.mul!(
  C::StridedVecOrMat,
  A::AbstractSparseMatrixCOO,
  B::SparseArrays.DenseInputVecOrMat,
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
  @eval function LinearAlgebra.mul!(
    C::StridedVecOrMat,
    xA::$T{<:Any, <:AbstractSparseMatrixCOO},
    B::SparseArrays.DenseInputVecOrMat,
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
  @eval function LinearAlgebra.mul!(
    C::StridedVecOrMat,
    xA::$T{<:Any, <:AbstractSparseMatrixCOO},
    B::SparseArrays.DenseInputVecOrMat,
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
