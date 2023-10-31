using LinearAlgebra, SparseArrays, FFTW, Random, Plots

include("sFOM.jl")
include("setupSketchingHandle.jl")
include("whitenBasis.jl")

# Seed for reproducability
Random.seed!(1);

# Define problem 
N = 100;
M = 350;

x0 = 0; 
x1 = 1;
h = (x1-x0)/N;

t0 = 0;
t1 = 0.02;
k = (t1-t0)/M;

# Initial condition
int_vec = (x0+h):h:(x1-h);
u0 = sin.(4*pi*int_vec);

# Allocate for solution
solmat = zeros(M+1, N-1);
solmat[1,:] = u0;

# Disc matW
rows = vcat(1:(N-1), 1:(N-2), 2:(N-1));
cols = vcat(1:(N-1), 2:(N-1) ,1:(N-2));
vals = vcat(-2*ones(N-1), ones(N-2), ones(N-2));
T = 1/h^2*sparse(rows, cols, vals);

# RHS
#G(U) = T*U  + 1 ./(1 .+ U.^2); # ostermann problem 
G(U) = T*U; # 1D heat eq.

# Jacobian 
#J(U) = T + sparse(1:(N-1), 1:(N-1), -2*U ./(1 .+ U.^2).^2); # ostermann problem 
J(U) = T; # 1D heat eq.

# φ-function
φ(X) = X\(exp(X)-sparse(I, size(X)));

# set params for sFOM
num_it = min(Int(round(0.4*N)), 100); 
trunc_len = 4;
mgs = true;
iter_diff_tol = 10^(-9);

# setup sketching
sketch = setupSketchingHandle(N-1, 2*num_it);

# do iters
for i in 1:M

    matfunceval, _, _ = sFOM(k*J(solmat[i,:]), G(solmat[i,:]), φ, num_it, trunc_len, mgs, iter_diff_tol, sketch);

    solmat[i+1,:] = solmat[i,:] + k*matfunceval;

end

# add BCs
solmat = hcat(zeros(M+1), solmat, zeros(M+1)); 

# plot sol
hm = heatmap(x0:h:x1, t0:k:t1, solmat,
            color=:thermal
            )
xlabel!("x")
ylabel!("t")
display(hm)
