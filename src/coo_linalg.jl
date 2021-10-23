function LinearAlgebra.mul!(C::StridedVecOrMat, A::AbstractSparseMatrixCOO, B::SparseArrays.DenseInputVecOrMat, α::Number, β::Number)
  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())
  size(B, 2) == size(C, 2) || throw(DimensionMismatch())
  if β != 1
    β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
  end
  @inbounds for k = 1:nnz(A)
    i, j = A.rows[k], A.cols[k]
    @views C[i, :] += α * A.vals[k] * B[j, :]
  end
  C
end

for (T, t) in ((Adjoint, adjoint), (Transpose, transpose))
  @eval function LinearAlgebra.mul!(C::StridedVecOrMat, xA::$T{<:Any,<:AbstractSparseMatrixCOO}, B::SparseArrays.DenseInputVecOrMat, α::Number, β::Number)
    A = xA.parent
    size(A, 2) == size(C, 1) || throw(DimensionMismatch())
    size(A, 1) == size(B, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    if β != 1
      β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    @inbounds for k = 1:nnz(A)
      j, i = A.rows[k], A.cols[k]
      @views C[i, :] += α * $t(A.vals[k]) * B[j, :]
    end
    C
  end
end

for (T, t) in ((Hermitian, adjoint), (Symmetric, transpose))
  @eval function LinearAlgebra.mul!(C::StridedVecOrMat, xA::$T{<:Any,<:AbstractSparseMatrixCOO}, B::SparseArrays.DenseInputVecOrMat, α::Number, β::Number)
    A = xA.data
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    if β != 1
      β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    @inbounds for k = 1:nnz(A)
      i, j, a = A.rows[k], A.cols[k], A.vals[k]
      (xA.uplo == :U && i < j) || (xA.uplo == :L && i > j) && continue  # ignore elements in this triangle
      @views C[i, :] += a * B[j, :]
      if i != j
        @views C[j, :] += $t(a) * B[i, :]
      end
    end
    C
  end
end

