using LinearAlgebra, SparseArrays, FFTW, Random, Plots

include("sFOM.jl")
include("setupSketchingHandle.jl")
include("whitenBasis.jl")

# Seed for reproducability
Random.seed!(1);

# define problem 
N = 100; # assume dx = dy
M = 250;

x0 = 0;
x1 = 1;
y0 = 0;
y1 = 1;
h = (x1-x0)/N;

xvec = (x0+h):h:(x1-h);
yvec = (y0+h):h:(y1-h);

t0 = 0;
t1 = 10;
k = (t1-t0)/M;

# initial condition
solvec = zeros((N-1)*(N-1));

# disc mat
rows = vcat(1:(N-1), 1:(N-2), 2:(N-1));
cols = vcat(1:(N-1), 2:(N-1) ,1:(N-2));
vals = vcat(-2ones(N-1), ones(N-2), ones(N-2));
T = 1/h^2*sparse(rows, cols, vals);

A = kron(sparse(I,N-1,N-1), T) + kron(T, sparse(I,N-1,N-1));

# RHS
f(x,y) = 20exp(-40sqrt((x-0.3)^2 + (y-0.3)^2)) + 20exp(-40sqrt((x-0.7)^2 + (y-0.7)^2)); # arbitrary sum of "point-sources"
source = [f(x,y) for x in xvec, y in yvec];
source = vec(source);
G(U) = A*U + source; # 2D heat eq.

# Jacobian
J(U) = A; # 2D heat eq.

# φ-function
φ(X) = X\(exp(X) - sparse(I, size(X)));


# set params for sFOM
num_it = min(Int(round(0.4*(N-1)*(N-1))), 400); 
trunc_len = 4;
mgs = true;
iter_diff_tol = 10^(-9);

# setup sketching
sketch = setupSketchingHandle((N-1)*(N-1), 2num_it);

# do iters
for i in 1:M

    global solvec;

    matfunceval, _, _ = sFOM(k*J(solvec), G(solvec), φ, num_it, trunc_len, mgs, iter_diff_tol, sketch);

    solvec = solvec + k*matfunceval;

end

solvec = Real.(solvec); # should i need to do this??

solmat = reshape(solvec, N-1, N-1);

# add BCs
solmat = hcat(zeros(N-1), solmat, zeros(N-1));
solmat = vcat(zeros(N+1)', solmat);
solmat = vcat(solmat, zeros(N+1)');

# plot sol
hm = heatmap(x0:h:x1, y0:h:y1, solmat,
            color=:thermal
            )
xlabel!("x")
ylabel!("y")
display(hm)





















