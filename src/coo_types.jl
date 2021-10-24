export rows, columns

"""
    AbstractSparseMatrixCOO{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}

Supertype for matrix in sparse coordinate format (COO).
"""
abstract type AbstractSparseMatrixCOO{Tv, Ti <: Integer} <: AbstractSparseMatrix{Tv, Ti} end

mutable struct SparseMatrixCOO{Tv, Ti <: Integer} <: AbstractSparseMatrixCOO{Tv, Ti}
  m::Int
  n::Int
  rows::Vector{Ti}
  cols::Vector{Ti}
  vals::Vector{Tv}

  function SparseMatrixCOO{Tv, Ti}(
    m::Integer,
    n::Integer,
    rows::Vector{Ti},
    cols::Vector{Ti},
    vals::Vector{Tv},
  ) where {Tv, Ti <: Integer}
    @noinline throwsz(str, lbl, k) =
      throw(ArgumentError("number of $str ($lbl) must be ≥ 0, got $k"))
    m < 0 && throwsz("rows", 'm', m)
    n < 0 && throwsz("columns", 'n', n)
    nnz = length(vals)
    @lencheck nnz rows cols
    new(Int(m), Int(n), rows, cols, vals)
  end
end

function SparseMatrixCOO(m::Integer, n::Integer, rows::Vector, cols::Vector, vals::Vector)
  Tv = eltype(vals)
  Ti = promote_type(eltype(rows), eltype(cols))
  SparseArrays.sparse_check_Ti(m, n, Ti)
  nnz = length(vals)
  @lencheck nnz rows cols
  # silently shorten rowval and nzval to usable index positions.
  maxlen = abs(widemul(m, n))
  isbitstype(Ti) && (maxlen = min(maxlen, typemax(Ti) - 1))
  length(rows) > maxlen && resize!(rows, maxlen)
  length(cols) > maxlen && resize!(cols, maxlen)
  length(vals) > maxlen && resize!(vals, maxlen)
  SparseMatrixCOO{Tv, Ti}(m, n, rows, cols, vals)
end

Base.size(A::SparseMatrixCOO) = (getfield(A, :m), getfield(A, :n))

"""
    nnz(A)

Returns the number of stored (filled) elements in a sparse array.

# Examples
```jldoctest
julia> A = sparse(2I, 3, 3)
3×3 SparseMatrixCSC{Int64,Int64} with 3 stored entries:
  [1, 1]  =  2
  [2, 2]  =  2
  [3, 3]  =  2
julia> nnz(A)
3
```
"""
SparseArrays.nnz(A::SparseMatrixCOO) = length(A.vals)
SparseArrays.nnz(A::Transpose{Tv, SparseMatrixCOO{Tv, Ti}}) where {Tv, Ti} = nnz(A.parent)
SparseArrays.nnz(A::Adjoint{Tv, SparseMatrixCOO{Tv, Ti}}) where {Tv, Ti} = nnz(A.parent)

"""
    nonzeros(A)

Return a vector of the structural nonzero values in sparse array `A`. This
includes zeros that are explicitly stored in the sparse array. The returned
vector points directly to the internal nonzero storage of `A`, and any
modifications to the returned vector will mutate `A` as well. See
[`rows`](@ref) and [`nzrange`](@ref).

# Examples
```jldoctest
julia> A = sparse(2I, 3, 3)
3×3 SparseMatrixCSC{Int64,Int64} with 3 stored entries:
  [1, 1]  =  2
  [2, 2]  =  2
  [3, 3]  =  2
julia> nonzeros(A)
3-element Array{Int64,1}:
 2
 2
 2
```
"""
SparseArrays.nonzeros(A::SparseMatrixCOO) = getfield(A, :vals)
Base.values(A::SparseMatrixCOO) = nonzeros(A)

"""
    rows(A::AbstractSparseMatrixCOO)

Return a vector of the row indices of `A`. Any modifications to the returned
vector will mutate `A` as well. Providing access to how the row indices are
stored internally can be useful in conjunction with iterating over structural
nonzero values. See also [`nonzeros`](@ref) and [`nzrange`](@ref).

# Examples
```jldoctest
julia> A = sparse(2I, 3, 3)
3×3 SparseMatrixCSC{Int64,Int64} with 3 stored entries:
  [1, 1]  =  2
  [2, 2]  =  2
  [3, 3]  =  2
julia> rowvals(A)
3-element Array{Int64,1}:
 1
 2
 3
```
"""
SparseArrays.rowvals(A::SparseMatrixCOO) = getfield(A, :rows)
rows(A::SparseMatrixCOO) = rowvals(A)
columns(A::SparseMatrixCOO) = getfield(A, :cols)

SparseArrays.findnz(A::AbstractSparseMatrixCOO{Tv, Ti}) where {Tv, Ti} =
  (rows(A), columns(A), values(A))
SparseArrays.findnz(A::Transpose{Tv, T}) where {Tv, T <: AbstractSparseMatrixCOO} =
  (columns(A.parent), rows(A.parent), values(A.parent))
SparseArrays.findnz(A::Adjoint{Tv, T}) where {Tv <: Real, T <: AbstractSparseMatrixCOO} =
  (columns(A.parent), rows(A.parent), values(A.parent))
SparseArrays.findnz(A::Adjoint{Tv, T}) where {Tv, T <: AbstractSparseMatrixCOO} =
  (columns(A.parent), rows(A.parent), ajoint(copy((values(A.parent)))))

# show

function Base.show(io::IO, ::MIME"text/plain", S::AbstractSparseMatrixCOO)
  xnnz = nnz(S)
  m, n = size(S)
  print(io, m, "×", n, " ", typeof(S), " with ", xnnz, " stored ", xnnz == 1 ? "entry" : "entries")
  if xnnz != 0
    print(io, ":\n")
    show(IOContext(io, :typeinfo => eltype(S)), S)
  end
end

function Base.show(
  io::IO,
  ::MIME"text/plain",
  S::Transpose{Tv, T},
) where {Tv, T <: AbstractSparseMatrixCOO}
  xnnz = nnz(S)
  m, n = size(S)
  print(io, m, "×", n, " ", typeof(S), " with ", xnnz, " stored ", xnnz == 1 ? "entry" : "entries")
  if xnnz != 0
    print(io, ":\n")
    show(IOContext(io, :typeinfo => eltype(S)), S)
  end
end

function Base.show(
  io::IO,
  ::MIME"text/plain",
  S::Adjoint{Tv, T},
) where {Tv, T <: AbstractSparseMatrixCOO}
  xnnz = nnz(S)
  m, n = size(S)
  print(io, m, "×", n, " ", typeof(S), " with ", xnnz, " stored ", xnnz == 1 ? "entry" : "entries")
  if xnnz != 0
    print(io, ":\n")
    show(IOContext(io, :typeinfo => eltype(S)), S)
  end
end

Base.show(io::IO, A::AbstractSparseMatrixCOO) =
  Base.show(convert(IOContext, io), A::AbstractSparseMatrixCOO)
Base.show(io::IO, A::Transpose{Tv, T}) where {Tv, T <: AbstractSparseMatrixCOO} =
  Base.show(convert(IOContext, io), A)
Base.show(io::IO, A::Adjoint{Tv, T}) where {Tv, T <: AbstractSparseMatrixCOO} =
  Base.show(convert(IOContext, io), A)

function Base.show(io::IOContext, A::AbstractSparseMatrixCOO)
  p = spy(size(A)..., findnz(A)..., title = "")
  show(io, p)
end

function Base.show(io::IOContext, A::Transpose{Tv, T}) where {Tv, T <: AbstractSparseMatrixCOO}
  p = spy(size(A)..., findnz(A)..., title = "")
  show(io, p)
end

function Base.show(io::IOContext, A::Adjoint{Tv, T}) where {Tv, T <: AbstractSparseMatrixCOO}
  p = spy(size(A)..., findnz(A)..., title = "")
  show(io, p)
end

Base.show(io::IOContext, A::Symmetric{Tv, SparseMatrixCOO{Tv, Ti}}) where {Tv, Ti <: Integer} =
  show(io, A.data)
Base.show(io::IOContext, A::Hermitian{Tv, SparseMatrixCOO{Tv, Ti}}) where {Tv, Ti <: Integer} =
  show(io, A.data)

# copy

Base.copy(A::AbstractSparseMatrixCOO) =
  SparseMatrixCOO(size(A)..., copy(rows(A)), copy(columns(A)), copy(values(A)))
SparseArrays.sparse(A::AbstractSparseMatrixCOO) = copy(A)

# convert to a SparseMatrixCOO

function SparseMatrixCOO(A::Matrix)
  I = findall(!iszero, A)
  SparseMatrixCOO(size(A)..., getindex.(I, 1), getindex.(I, 2), A[I])
end

SparseMatrixCOO(A::SparseMatrixCSC) = SparseMatrixCOO(size(A)..., findnz(A)...)

# indexing

Base.getindex(A::AbstractSparseMatrixCOO, I::Tuple{Integer, Integer}) = Base.getindex(A, I[1], I[2])

function Base.getindex(A::AbstractSparseMatrixCOO{Tv, Ti}, i0::Integer, i1::Integer) where {Tv, Ti}
  m, n = size(A)
  (1 ≤ i0 ≤ m && 1 ≤ i1 ≤ n) || throw(BoundsError())
  val = zero(Tv)
  for k = 1:nnz(A)
    i, j, v = A.rows[k], A.cols[k], A.vals[k]
    if i == i0 && j == i1
      val += v
    end
  end
  return val
end

# converting from SparseMatrixCOO to other matrix types

_goodbuffers(S::SparseMatrixCOO) = _goodbuffers(size(S)..., findnz(S)...)
_checkbuffers(S::SparseMatrixCOO) = (@assert _goodbuffers(S); S)
_checkbuffers(S::Union{Adjoint, Transpose}) = (_checkbuffers(parent(S)); S)

function _goodbuffers(m, n, rows, cols, vals)
  (length(rows) == length(cols) == length(vals)) && all(1 .≤ rows .≤ m) && all(1 .≤ cols .≤ n)
end

function Matrix(S::AbstractSparseMatrixCOO{Tv}) where {Tv}
  _checkbuffers(S)
  A = Matrix{Tv}(undef, size(S)...)
  fill!(A, zero(Tv))
  for k = 1:nnz(S)
    i, j, v = S.rows[k], S.cols[k], S.vals[k]
    A[i, j] += v
  end
  return A
end
Array(S::AbstractSparseMatrixCOO) = Matrix(S)

convert(T::Type{<:AbstractSparseMatrixCOO}, m::AbstractMatrix) = m isa T ? m : T(m)

convert(T::Type{<:Diagonal}, m::AbstractSparseMatrixCOO) =
  m isa T ? m : isdiag(m) ? T(m) : throw(ArgumentError("matrix cannot be represented as Diagonal"))
convert(T::Type{<:SymTridiagonal}, m::AbstractSparseMatrixCOO) =
  m isa T ? m :
  issymmetric(m) && isbanded(m, -1, 1) ? T(m) :
  throw(ArgumentError("matrix cannot be represented as SymTridiagonal"))
convert(T::Type{<:Tridiagonal}, m::AbstractSparseMatrixCOO) =
  m isa T ? m :
  isbanded(m, -1, 1) ? T(m) : throw(ArgumentError("matrix cannot be represented as Tridiagonal"))
convert(T::Type{<:LowerTriangular}, m::AbstractSparseMatrixCOO) =
  m isa T ? m :
  istril(m) ? T(m) : throw(ArgumentError("matrix cannot be represented as LowerTriangular"))
convert(T::Type{<:UpperTriangular}, m::AbstractSparseMatrixCOO) =
  m isa T ? m :
  istriu(m) ? T(m) : throw(ArgumentError("matrix cannot be represented as UpperTriangular"))

float(S::SparseMatrixCOO) =
  SparseMatrixCOO(size(S, 1), size(S, 2), copy(rows(S)), copy(columns(S)), float.(nonzeros(S)))
complex(S::SparseMatrixCOO) = SparseMatrixCOO(
  size(S, 1),
  size(S, 2),
  copy(rows(S)),
  copy(columns(S)),
  complex(copy(nonzeros(S))),
)
