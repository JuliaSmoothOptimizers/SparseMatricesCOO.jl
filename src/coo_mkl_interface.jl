for T in (Float32, Float64, ComplexF32, ComplexF64)

  op_wrappers = ((identity                     , 'N', identity          ),
                 (M -> :(Transpose{<:$T, <:$M}), 'T', A -> :(parent($A))),
                 (M -> :(Adjoint{<:$T, <:$M})  , 'C', A -> :(parent($A)))
  )

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

      function LinearAlgebra.:(*)(A::$TypeA, B::StridedMatrix{$T})
        m, n = size(A)
        k, p = size(B)
        n == k || throw(DimensionMismatch())
        C = Matrix{$T}(undef, m, p)
        mul!(C, A, B)
      end
    end
  end
end
