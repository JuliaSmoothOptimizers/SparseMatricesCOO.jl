import LinearAlgebra.BlasInt

matdescra(A::LowerTriangular) = "TLNF"
matdescra(A::UpperTriangular) = "TUNF"
matdescra(A::Diagonal) = "DUNF"
matdescra(A::UnitLowerTriangular) = "TLUF"
matdescra(A::UnitUpperTriangular) = "TUUF"
matdescra(A::Symmetric) = string('S', A.uplo, 'N', 'F')
matdescra(A::Hermitian) = string('H', A.uplo, 'N', 'F')
matdescra(A::SparseMatrixCOO) = "GUUF"

function check_transa(transa::Char)
  transa == 'N' || transa == 'T' || transa == 'C' || error("transa is '$transa', must be 'N', 'T', or 'C'")
end

function check_uplo(uplo::Char)
  uplo == 'L' ||  uplo == 'U' || error("uplo is '$uplo', must be L' or 'U'")
end

function check_diag(diag::Char)
  diag == 'U' ||  diag == 'N' || error("diag is '$diag', must be U' or 'N'")
end

for (mv, sv, symv, trsv, mm, sm, T) in (("mkl_scoomv", "mkl_scoosv", "mkl_scoosymv", "mkl_scootrmv", "mkl_scoomm", "mkl_scoosm", :Float32),
                                        ("mkl_dcoomv", "mkl_dcoosv", "mkl_dcoosymv", "mkl_dcootrmv", "mkl_dcoomm", "mkl_dcoosm", :Float64),
                                        ("mkl_ccoomv", "mkl_ccoosv", "mkl_ccoosymv", "mkl_ccootrmv", "mkl_ccoomm", "mkl_ccoosm", :ComplexF32),
                                        ("mkl_zcoomv", "mkl_zcoosv", "mkl_zcoosymv", "mkl_zcootrmv", "mkl_zcoomm", "mkl_zcoosm", :ComplexF64))

  @eval begin
    function coomv!(transa::Char, alpha::$T, matdescra::String, A::SparseMatrixCOO{$T, BlasInt}, x::StridedVector{$T}, beta::$T, y::StridedVector{$T})
      check_transa(transa)
      ccall(($mv, libmkl_rt),
             Cvoid,
            (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{$T}, Ptr{UInt8}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{$T}, Ptr{$T}),
             transa    , A.m         , A.n         , alpha  , matdescra , A.vals , A.rows      , A.cols      , nnz(A)      , x      , beta   , y      )
      return y
    end

    function coomm!(transa::Char, alpha::$T, matdescra::String, A::SparseMatrixCOO{$T, BlasInt}, B::StridedMatrix{$T}, beta::$T, C::StridedMatrix{$T})
      check_transa(transa)
      mB, nB = size(B)
      mC, nC = size(C)
      ccall(($mm, libmkl_rt),
             Cvoid,
            (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{$T}, Ptr{UInt8}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ref{$T}, Ptr{$T}, Ref{BlasInt}),
             transa    , A.m         , nC          , A.n         , alpha  , matdescra , A.vals , A.rows      , A.cols      , nnz(A)      , B      , mB          , beta   , C      , mC          )
      return C
    end

    function coosymv!(uplo::Char, A::SparseMatrixCOO{$T, BlasInt}, x::StridedVector{$T}, y::StridedVector{$T})
      checksquare(A)
      check_uplo(transa)
      ccall(($symv, libmkl_rt),
             Cvoid,
            (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ptr{$T}),
             transa    , A.m         , A.vals , A.rows      , A.cols      , nnz(A)      , x      , y      )
      return y
    end

    function cootrsv!(uplo::Char, transa::Char, diag::Char, A::SparseMatrixCOO{$T, BlasInt}, x::StridedVector{$T}, y::StridedVector{$T})
      checksquare(A)
      check_uplo(transa)
      check_transa(transa)
      check_diag(transa)
      ccall(($trsv, libmkl_rt),
             Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ptr{$T}),
             uplo      , transa    , diag      , A.m         , A.vals , A.rows      , A.cols      , nnz(A)      , x      , y      )
      return y
    end

    function coosv!(transa::Char, alpha::$T, matdescra::String, A::SparseMatrixCOO{$T, BlasInt}, x::StridedVector{$T}, y::StridedVector{$T})
      checksquare(A)
      check_transa(transa)
      ccall(($sv, libmkl_rt),
             Cvoid,
            (Ref{UInt8}, Ref{BlasInt}, Ref{$T}, Ptr{UInt8}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ptr{$T}),
             transa    , A.m         , alpha  , matdescra , A.vals , A.rows      , A.cols      , nnz(A)      , x      , y      )
      return y
    end

    function coosm!(transa::Char, alpha::$T, matdescra::String, A::SparseMatrixCOO{$T, BlasInt}, B::StridedMatrix{$T}, C::StridedMatrix{$T})
      checksquare(A)
      check_transa(transa)
      mB, nB = size(B)
      mC, nC = size(C)
      ccall(($sm, libmkl_rt),
            Cvoid,
           (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{$T}, Ptr{UInt8}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}),
            transa    , A.n         , nC          , alpha  , matdescra , A.vals , A.rows      , A.cols      , nnz(A)      , B      , mB          , C      , mC          )
      return C
    end
  end
end
