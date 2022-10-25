# @testset "CSC -- COO conversions $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
#   A_csc = sprand(T, 10, 20, 0.2)
#   A_coo = SparseMatrixCOO(A_csc)
#   A_csc2 = SparseMatrixCSC(A_coo)
#   @test A_csc ≈ A_csc2
# end

@testset "Generic mul! -- Vector $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for opa in [identity, transpose, adjoint]
    A_csc = sprand(T, 10, 10, 0.2)
    A_coo = SparseMatrixCOO(A_csc)
    x = rand(T, 10)
    y_csc = zeros(T, 10)
    y_coo = zeros(T, 10)
    mul!(y_csc, opa(A_csc), x)
    mul!(y_coo, opa(A_coo), x)
    @test y_csc ≈ y_coo
  end
end

@testset "Generic mul! -- Matrix $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for opa in [identity, transpose, adjoint]
    for opb in [identity, transpose, adjoint]
      A_csc = sprand(T, 10, 10, 0.2)
      A_coo = SparseMatrixCOO(A_csc)
      X = opb == identity ? rand(T, 10, 2) : rand(T, 2, 10)
      Y_csc = zeros(T, 10, 2)
      Y_coo = zeros(T, 10, 2)
      mul!(Y_csc, opa(A_csc), opb(X))
      mul!(Y_coo, opa(A_coo), opb(X))
      @test Y_csc ≈ Y_coo
    end
  end
end

@testset "Symmetric/Hermitian mul! -- Vector $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for wrappera in [Symmetric, Hermitian]
    wrappera == Hermitian && T <: Complex && continue
    A_csc = sprand(T, 10, 10, 0.2)
    A_csc = A_csc + A_csc'
    A_coo = SparseMatrixCOO(A_csc)
    x = rand(T, 10)
    y_csc = zeros(T, 10)
    y_coo = zeros(T, 10)
    mul!(y_csc, wrappera(A_csc), x)
    mul!(y_coo, wrappera(A_coo), x)
    @test y_csc ≈ y_coo
  end
end

@testset "Symmetric/Hermitian mul! -- Matrix $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for wrappera in [Symmetric, Hermitian]
    wrappera == Hermitian && T <: Complex && continue
    A_csc = sprand(T, 10, 10, 0.2)
    A_csc = A_csc + A_csc'
    A_coo = SparseMatrixCOO(A_csc)
    X = rand(T, 10, 2)
    Y_csc = zeros(T, 10, 2)
    Y_coo = zeros(T, 10, 2)
    mul!(Y_csc, wrappera(A_csc), X)
    mul!(Y_coo, wrappera(A_coo), X)
    @test Y_csc ≈ Y_coo
  end
end

@testset "SparseMatrixCOO * Vector $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for opa in [identity, transpose, adjoint]
    A_csc = sprand(T, 10, 10, 0.2)
    A_coo = SparseMatrixCOO(A_csc)
    x = rand(T, 10)
    y_csc = opa(A_csc) * x
    y_coo = opa(A_coo) * x
    @test y_csc ≈ y_coo
  end

  for wrappera in [Symmetric, Hermitian]
    wrappera == Hermitian && T <: Complex && continue
    A_csc = sprand(T, 10, 10, 0.2)
    A_csc = A_csc + A_csc'
    A_coo = SparseMatrixCOO(A_csc)
    x = rand(T, 10)
    y_csc = A_csc * x
    y_coo = A_coo * x
    @test y_csc ≈ y_coo
  end
end

@testset "SparseMatrixCOO * Matrix $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for opa in [identity, transpose, adjoint]
    for opb in [identity, transpose, adjoint]
      A_csc = sprand(T, 10, 10, 0.2)
      A_coo = SparseMatrixCOO(A_csc)
      X = opb == identity ? rand(T, 10, 2) : rand(T, 2, 10)
      Y_csc = opa(A_csc) * opb(X)
      Y_coo = opa(A_coo) * opb(X)
      @test Y_csc ≈ Y_coo
    end
  end

  for wrappera in [Symmetric, Hermitian]
    wrappera == Hermitian && T <: Complex && continue
    A_csc = sprand(T, 10, 10, 0.2)
    A_csc = A_csc + A_csc'
    A_coo = SparseMatrixCOO(A_csc)
    X = rand(T, 10, 2)
    Y_csc = A_csc * X
    Y_coo = A_coo * X
    @test Y_csc ≈ Y_coo
  end
end

@testset "ldiv! -- Vector $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for triangle in [LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular]
    for opa in [identity, transpose, adjoint]
      A_csc = sparse(rand(T, 10, 10))
      A_coo = SparseMatrixCOO(A_csc)
      y = rand(T, 10)
      x_csc = zeros(T, 10)
      x_coo = zeros(T, 10)
      ldiv!(x_csc, triangle(opa(A_csc)), y)
      ldiv!(x_coo, triangle(opa(A_coo)), y)
      @test x_csc ≈ x_coo
    end
  end
end

@testset "ldiv! -- Matrix $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for triangle in [LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular]
    for opa in [identity, transpose, adjoint]
      A_csc = sparse(rand(T, 10, 10))
      A_coo = SparseMatrixCOO(A_csc)
      Y = rand(T, 10, 2)
      X_csc = zeros(T, 10, 2)
      X_coo = zeros(T, 10, 2)
      ldiv!(X_csc, triangle(opa(A_csc)), Y)
      ldiv!(X_coo, triangle(opa(A_coo)), Y)
      @test X_csc ≈ X_coo
    end
  end
end
