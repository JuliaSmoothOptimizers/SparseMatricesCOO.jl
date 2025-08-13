# Everything in here is heavily borrowed from SparseArrays.jl.

const AbstractSparseMatrixCOOInclAdjointAndTranspose = Union{
  AbstractSparseMatrixCOO,
  Adjoint{<:Any, <:AbstractSparseMatrixCOO},
  Transpose{<:Any, <:AbstractSparseMatrixCOO},
}

function Base.isstored(A::AbstractSparseMatrixCOO, i::Integer, j::Integer)
  @boundscheck checkbounds(A, i, j)
  for k ∈ 1:nnz(A)
    (i == A.rows[k] && j == A.cols[k]) && return true
  end
  return false
end

Base.isstored(
  A::LinearAlgebra.AdjOrTrans{<:Any, <:AbstractSparseMatrixCOO},
  i::Integer,
  j::Integer,
) = isstored(parent(A), j, i)

Base.replace_in_print_matrix(
  A::AbstractSparseMatrixCOOInclAdjointAndTranspose,
  i::Integer,
  j::Integer,
  s::AbstractString,
) = Base.isstored(A, i, j) ? s : Base.replace_with_centered_mark(s)

function Base.array_summary(
  io::IO,
  S::AbstractSparseMatrixCOOInclAdjointAndTranspose,
  dims::Tuple{Vararg{Base.OneTo}},
)
  _checkbuffers(S)
  xnnz = nnz(S)
  m, n = size(S)
  print(io, m, "×", n, " ", typeof(S), " with ", xnnz, " stored ", xnnz == 1 ? "entry" : "entries")
  nothing
end

# called by `show(io, MIME("text/plain"), ::AbstractSparseMatrixCSCInclAdjointAndTranspose)`
function Base.print_array(io::IO, S::AbstractSparseMatrixCOOInclAdjointAndTranspose)
  if max(size(S)...) < 16
    Base.print_matrix(io, S)
  else
    _show_with_braille_patterns(io, S)
  end
end

# always show matrices as `sparse(I, J, K)`
function Base.show(io::IO, _S::AbstractSparseMatrixCOOInclAdjointAndTranspose)
  _checkbuffers(_S)
  S = _S isa Adjoint || _S isa Transpose ? _S.parent : _S
  I = rowvals(S)
  J = columns(S)
  K = nonzeros(S)
  m, n = size(S)
  if _S isa Adjoint
    print(io, "adjoint(")
  elseif _S isa Transpose
    print(io, "transpose(")
  end
  print(io, "sparse(", I, ", ", J, ", ", K, ", ", m, ", ", n, ")")
  if _S isa Adjoint || _S isa Transpose
    print(io, ")")
  end
end

Base.show(io::IOContext, A::Symmetric{Tv, SparseMatrixCOO{Tv, Ti}}) where {Tv, Ti <: Integer} =
  show(io, A.data)
Base.show(io::IOContext, A::Hermitian{Tv, SparseMatrixCOO{Tv, Ti}}) where {Tv, Ti <: Integer} =
  show(io, A.data)

const brailleBlocks = UInt16['⠁', '⠂', '⠄', '⡀', '⠈', '⠐', '⠠', '⢀']

function _show_with_braille_patterns(io::IO, S::AbstractSparseMatrixCOOInclAdjointAndTranspose)
  m, n = size(S)
  (m == 0 || n == 0) && return show(io, MIME("text/plain"), S)

  # The maximal number of characters we allow to display the matrix
  local maxHeight::Int, maxWidth::Int
  maxHeight = displaysize(io)[1] - 4 # -4 from [Prompt, header, newline after elements, new prompt]
  maxWidth = displaysize(io)[2] ÷ 2

  # In the process of generating the braille pattern to display the nonzero
  # structure of `S`, we need to be able to scale the matrix `S` to a
  # smaller matrix with the same aspect ratio as `S`, but fits on the
  # available screen space. The size of that smaller matrix is stored
  # in the variables `scaleHeight` and `scaleWidth`. If no scaling is needed,
  # we can use the size `m × n` of `S` directly.
  # We determine if scaling is needed and set the scaling factors
  # `scaleHeight` and `scaleWidth` accordingly. Note that each available
  # character can contain up to 4 braille dots in its height (⡇) and up to
  # 2 braille dots in its width (⠉).
  if get(io, :limit, true) && (m > 4maxHeight || n > 2maxWidth)
    s = min(2maxWidth / n, 4maxHeight / m)
    scaleHeight = floor(Int, s * m)
    scaleWidth = floor(Int, s * n)
  else
    scaleHeight = m
    scaleWidth = n
  end

  # Make sure that the matrix size is big enough to be able to display all
  # the corner border characters
  if scaleHeight < 8
    scaleHeight = 8
  end
  if scaleWidth < 4
    scaleWidth = 4
  end

  # `brailleGrid` is used to store the needed braille characters for
  # the matrix `S`. Each row of the braille pattern to print is stored
  # in a column of `brailleGrid`.
  brailleGrid = fill(UInt16(10240), (scaleWidth - 1) ÷ 2 + 4, (scaleHeight - 1) ÷ 4 + 1)
  brailleGrid[1, :] .= '⎢'
  brailleGrid[end - 1, :] .= '⎥'
  brailleGrid[1, 1] = '⎡'
  brailleGrid[1, end] = '⎣'
  brailleGrid[end - 1, 1] = '⎤'
  brailleGrid[end - 1, end] = '⎦'
  brailleGrid[end, :] .= '\n'

  rowscale = max(1, scaleHeight - 1) / max(1, m - 1)
  colscale = max(1, scaleWidth - 1) / max(1, n - 1)
  @inbounds for k ∈ 1:nnz(S)
    if isa(S, AbstractSparseMatrixCOO)
      j = S.cols[k]
      i = S.rows[k]
    else
      # If `S` is a adjoint or transpose of a sparse matrix we invert the
      # roles of the indices `i` and `j`
      i = parent(S).cols[k]
      j = parent(S).rows[k]
    end
    # Scale the column index `j` to the best matching column index
    # of a matrix of size `scaleHeight × scaleWidth`
    sj = round(Int, (j - 1) * colscale + 1)
    # Scale the row index `i` to the best matching row index
    # of a matrix of size `scaleHeight × scaleWidth`
    si = round(Int, (i - 1) * rowscale + 1)

    # Given the index pair `(si, sj)` of the scaled matrix,
    # calculate the corresponding triple `(k, l, p)` such that the
    # element at `(si, sj)` can be found at position `(k, l)` in the
    # braille grid `brailleGrid` and corresponds to the 1-dot braille
    # character `brailleBlocks[p]`
    k = (sj - 1) ÷ 2 + 2
    l = (si - 1) ÷ 4 + 1
    p = ((sj - 1) % 2) * 4 + ((si - 1) % 4 + 1)

    brailleGrid[k, l] |= brailleBlocks[p]
  end
  foreach(c -> print(io, Char(c)), @view brailleGrid[1:(end - 1)])
end
