struct DimensionError <: Exception
  name :: Union{Symbol,String}
  dim_expected :: Int
  dim_found :: Int
end

function Base.showerror(io::IO, e::DimensionError)
  print(io, "DimensionError: Input $(e.name) should have length $(e.dim_expected) not $(e.dim_found)")
end

# Check that arrays have a prescribed size.
# https://groups.google.com/forum/?fromgroups=#!topic/julia-users/b6RbQ2amKzg
macro lencheck(l, vars...)
  exprs = Expr[]
  for var in vars
    varname = string(var)
    push!(exprs,
          :(if length($(esc(var))) != $(esc(l))
                throw(DimensionError($varname, $(esc(l)), length($(esc(var)))))
            end))
  end
  Expr(:block, exprs...)
end
