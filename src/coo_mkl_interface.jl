for T in (Float32, Float64, ComplexF32, ComplexF64)

  op_wrappers = ((identity                 , 'N', identity          ),
                 (M -> :(Transpose{$T, $M}), 'T', A -> :(parent($A))),
                 (M -> :(Adjoint{$T, $M})  , 'C', A -> :(parent($A))))

  for (wrapa, transa, unwrapa) in op_wrappers
    TypeA = wrapa(:(SparseMatrixCOO{$T, $BlasInt}))
    
    @eval begin
      function LinearAlgebra.mul!(y::StridedVector{$T}, A::$TypeA, x::StridedVector{$T}, alpha::Number, beta::Number)
        return coomv!($transa, $T(alpha), "GUUF", $(unwrapa(:A)), x, $T(beta), y)
      end

      function LinearAlgebra.mul!(C::StridedMatrix{$T}, A::$TypeA, B::StridedMatrix{$T}, alpha::Number, beta::Number)
        return coomm!($transa, $T(alpha), "GUUF", $(unwrapa(:A)), B, $T(beta), C)
      end

      LinearAlgebra.mul!(y::StridedVector{$T}, A::$TypeA, x::StridedVector{$T}) = mul!(y, A, x, one($T), zero($T))

      LinearAlgebra.mul!(C::StridedMatrix{$T}, A::$TypeA, B::StridedMatrix{$T}) = mul!(C, A, B, one($T), zero($T))

      function LinearAlgebra.:(*)(A::$TypeA, x::StridedVector{$T})
        m, n = size(A)
        length(x) == n || throw(DimensionMismatch())
        y = Vector{$T}(undef, m)
        mul!(y, A, x)
      end
    end
  end

  op_symmetric = (M -> :(Symmetric{$T, $M}), A -> :(parent($A)))
  op_hermitian = (M -> :(Hermitian{$T, $M}), A -> :(parent($A)))
  op_wrappers = T <: Real ? (op_symmetric, op_hermitian) : (op_symmetric,)

  for (wrapa, unwrapa) in op_wrappers
    TypeA = wrapa(:(SparseMatrixCOO{$T, $BlasInt}))

    @eval begin
      LinearAlgebra.mul!(y::StridedVector{$T}, A::$TypeA, x::StridedVector{$T}) = coosymv!(A.uplo, $(unwrapa(:A)), x, y)

      function LinearAlgebra.:(*)(A::$TypeA, x::StridedVector{$T})
        m, n = size(A)
        length(x) == n || throw(DimensionMismatch())
        y = Vector{$T}(undef, m)
        mul!(y, parent(A), x)
      end
    end
  end
end

for (triangle, matdescra) in ((:LowerTriangular, "TLNF"),
                              (:UnitLowerTriangular, "TLUF"),
                              (:UpperTriangular, "TUNF"),
                              (:UnitUpperTriangular, "TUUF"))

  @eval begin
    LinearAlgebra.ldiv!(x::StridedVector{T},
                        A::$triangle{T,SparseMatrixCOO{T, BlasInt}},
                        y::StridedVector{T}) where {T <: BlasFloat} =
      coosv!('N', one(T), $matdescra, parent(A), x, y)

    LinearAlgebra.ldiv!(X::StridedMatrix{T},
                        A::$triangle{T,SparseMatrixCOO{T, BlasInt}},
                        Y::StridedMatrix{T}) where {T <: BlasFloat} =
      coosm!('N', one(T), $matdescra, parent(A), X, Y)
  end
end

for (triangle, matdescra) in ((:LowerTriangular, "TUNF"),
                              (:UnitLowerTriangular, "TUUF"),
                              (:UpperTriangular, "TLNF"),
                              (:UnitUpperTriangular, "TLUF"))

  for (opa, transa) in ((:Transpose, 'T'),
                        (:Adjoint, 'C'))
    @eval begin
      LinearAlgebra.ldiv!(x::StridedVector{T},
                          A::$triangle{T,$opa{T,SparseMatrixCOO{T, BlasInt}}},
                          y::StridedVector{T}) where {T <: BlasFloat} =
        coosv!($transa, one(T), $matdescra, parent(parent(A)), x, y)

      LinearAlgebra.ldiv!(x::StridedMatrix{T},
                          A::$triangle{T,$opa{T,SparseMatrixCOO{T, BlasInt}}},
                          y::StridedMatrix{T}) where {T <: BlasFloat} =
        coosm!($transa, one(T), $matdescra, parent(parent(A)), x, y)
    end
  end
end

# SparseMatrixCOO(A::SparseMatrixCSC{T, BlasInt}) where {T <: BlasFloat} = csc_coo(A)
# SparseMatrixCSC(A::SparseMatrixCOO{T, BlasInt}) where {T <: BlasFloat} = coo_csc(A)
