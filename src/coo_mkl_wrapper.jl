import LinearAlgebra: BlasInt, BlasFloat, BlasReal, BlasComplex, checksquare

# matdescra(A::LowerTriangular) = "TLNF"
# matdescra(A::UpperTriangular) = "TUNF"
# matdescra(A::Diagonal) = "DUNF"
# matdescra(A::UnitLowerTriangular) = "TLUF"
# matdescra(A::UnitUpperTriangular) = "TUUF"
# matdescra(A::Symmetric) = string('S', A.uplo, 'N', 'F')
# matdescra(A::Hermitian) = string('H', A.uplo, 'N', 'F')
# matdescra(A::SparseMatrixCOO) = "GUUF"

#! format: off
function check_transa(transa::Char)
  transa == 'N' || transa == 'T' || transa == 'C' || error("transa is '$transa', must be 'N', 'T', or 'C'")
end

function check_uplo(uplo::Char)
  uplo == 'L' ||  uplo == 'U' || error("uplo is '$uplo', must be L' or 'U'")
end

for (mv, sv, symv, mm, sm, csrcoo, T) in (("mkl_scoomv", "mkl_scoosv", "mkl_scoosymv", "mkl_scoomm", "mkl_scoosm", "mkl_scsrcoo", :Float32),
                                          ("mkl_dcoomv", "mkl_dcoosv", "mkl_dcoosymv", "mkl_dcoomm", "mkl_dcoosm", "mkl_dcsrcoo", :Float64),
                                          ("mkl_ccoomv", "mkl_ccoosv", "mkl_ccoosymv", "mkl_ccoomm", "mkl_ccoosm", "mkl_ccsrcoo", :ComplexF32),
                                          ("mkl_zcoomv", "mkl_zcoosv", "mkl_zcoosymv", "mkl_zcoomm", "mkl_zcoosm", "mkl_zcsrcoo", :ComplexF64))

  @eval begin
    function coomv!(transa::Char, alpha::$T, matdescra::String, A::SparseMatrixCOO{$T, BlasInt}, x::StridedVector{$T}, beta::$T, y::StridedVector{$T})
      check_transa(transa)
      ccall(($mv, libmkl_rt),
             Cvoid,
            (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{$T}, Ptr{UInt8}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{$T}, Ptr{$T}),
             transa    , A.m         , A.n         , alpha  , matdescra , A.vals , A.rows      , A.cols      , nnz(A)      , x      , beta   , y      )
      return y
    end

    function coosv!(transa::Char, alpha::$T, matdescra::String, A::SparseMatrixCOO{$T, BlasInt}, x::StridedVector{$T}, y::StridedVector{$T})
      checksquare(A)
      check_transa(transa)
      ccall(($sv, libmkl_rt),
             Cvoid,
            (Ref{UInt8}, Ref{BlasInt}, Ref{$T}, Ptr{UInt8}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ptr{$T}),
             transa    , A.m         , alpha  , matdescra , A.vals , A.rows      , A.cols      , nnz(A)      , y      , x      )
      return y
    end

    function coosymv!(uplo::Char, A::SparseMatrixCOO{$T, BlasInt}, x::StridedVector{$T}, y::StridedVector{$T})
      checksquare(A)
      check_uplo(uplo)
      ccall(($symv, libmkl_rt),
             Cvoid,
            (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ptr{$T}),
             uplo      , A.m         , A.vals , A.rows      , A.cols      , nnz(A)      , x      , y      )
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

    function coosm!(transa::Char, alpha::$T, matdescra::String, A::SparseMatrixCOO{$T, BlasInt}, B::StridedMatrix{$T}, C::StridedMatrix{$T})
      checksquare(A)
      check_transa(transa)
      mB, nB = size(B)
      mC, nC = size(C)
      ccall(($sm, libmkl_rt),
            Cvoid,
           (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{$T}, Ptr{UInt8}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}),
            transa    , A.n         , nC          , alpha  , matdescra , A.vals , A.rows      , A.cols      , nnz(A)      , C      , mC          , B      , mB          )
      return C
    end

    function coo_csc(A::SparseMatrixCOO{$T, BlasInt})
      m,n = size(A)
      nnzA = nnz(A)
      colptr = Vector{BlasInt}(undef, n+1)
      rowval = Vector{BlasInt}(undef, nnzA)
      nzval  = Vector{$T}(undef, nnzA)
      info = Ref{BlasInt}()
      job = BlasInt[2, 1, 1, 0, nnzA, 0]

      ccall(($csrcoo, libmkl_rt),
            Cvoid,
           (Ptr{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            job         , m           , nnzA        , nzval  , rowval      , colptr      , A.vals , A.cols      , A.rows      , info        )

      info[] == 0 || error("The routine is interrupted because there is not enough space")
      return SparseMatrixCSC(colptr, rowval, nzval, m, n)
    end

    function csc_coo(A::SparseMatrixCSC{$T, BlasInt})
      m,n = size(A)
      nnzA = nnz(A)
      rows = Vector{BlasInt}(undef, nnzA)
      cols = Vector{BlasInt}(undef, nnzA)
      vals = Vector{$T}(undef, nnzA)
      info = Ref{BlasInt}()
      job = BlasInt[0, 1, 1, 0, nnzA, 3]

      ccall(($csrcoo, libmkl_rt),
            Cvoid,
           (Ptr{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            job         , m           , nnzA        , A.nzval, A.rowval    , A.colptr    , vals   , cols        , rows        , info        )

      info[] == 0 || error("The routine is interrupted because there is not enough space")
      return SparseMatrixCOO(m, n, rows, cols, vals)
    end
  end
end
#! format: on
