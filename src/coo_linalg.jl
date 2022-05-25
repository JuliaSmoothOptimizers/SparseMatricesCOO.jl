function mulCOO!(C::AbstractVector, Arows, Acols, Avals, B::AbstractVector, α, Annz)
  @inbounds for k = 1:Annz
    i, j = Arows[k], Acols[k]
    C[i] += α * Avals[k] * B[j]
  end
end

function mulCOO!(C::AbstractMatrix, Arows, Acols, Avals, B::AbstractMatrix, α, Annz)
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
  mulCOO!(C, A.rows, A.cols, A.vals, B, α, nnz(A))
  C
end

function mulCOOt!(C::AbstractVector, Arows, Acols, Avals, B::AbstractVector, α, Annz, t)
  @inbounds for k = 1:Annz
    i, j = Acols[k], Arows[k] 
    C[i] += α * t(Avals[k]) * B[j]
  end
end

function mulCOOt!(C::AbstractMatrix, Arows, Acols, Avals, B::AbstractMatrix, α, Annz, t)
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
    mulCOOt!(C, A.rows, A.cols, A.vals, B, α, nnz(A), $t)
    C
  end
end

function mulCOOsym!(C::AbstractVector, Arows, Acols, Avals, B::AbstractVector, α, Annz, t, uplo)
  @inbounds for k = 1:Annz
    i, j, a = Arows[k], Acols[k], Avals[k]
    ((uplo == 'U' && i > j) || (uplo == 'L' && i < j)) && continue  # ignore elements in this triangle
    C[i] += α * a * B[j]
    if i != j
      C[j] += α * t(a) * B[i]
    end
  end
end

function mulCOOsym!(C::AbstractMatrix, Arows, Acols, Avals, B::AbstractMatrix, α, Annz, t, uplo)
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
    mulCOOsym!(C, A.rows, A.cols, A.vals, B, α, nnz(A), $t, xA.uplo)
    C
  end
end
