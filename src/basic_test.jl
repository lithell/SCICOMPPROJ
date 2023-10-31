using LinearAlgebra, SparseArrays, FFTW, Random, MatrixDepot, Plots, Printf, LaTeXStrings

include("gen_err_vec.jl")
include("setupSketchingHandle.jl")
include("whitenBasis.jl")


# Set up problem 
nn = 25;
A = matrixdepot("wathen", nn, nn);
A = -A;
N = size(A,1);

@printf "Problem size: %d\n" N

# sFOM params
num_it = 120;
trunc_len = 4;
mgs = true;

# Set up sketching
sketch_param = 2*num_it;
sketch = setupSketchingHandle(N, sketch_param);

# ex
f(x) = exp(x);

num_runs = 50;

p = plot();

f_mat = f(Matrix(A));

err_mat = zeros(num_it, num_runs);

for i = 1:num_runs

    # Seed for reproducability
    Random.seed!(i);
    
    b = 10*rand(N);
    ex = f_mat*b;

    # Do sFOM
    err_vec = gen_err_vec(A, b, f, num_it, trunc_len, mgs, sketch, ex);

    err_mat[:,i] = err_vec;

    global p = plot!(1:num_it, err_vec, 
                    yaxis=:log,
                    label=:none,
                    minorticks=:true,
                    linealpha=0.2,
                    lw=0.5,
                    lc=:purple,
                    #markershape=:+,
                    #markersize=2,
                    #mc=:red
                    ylimits=(10.0^-16, 10.0^0+1),
                    yticks=10.0 .^(-16:2:0)
                    );

end


worst_run = maximum(err_mat, dims=2);
best_run = minimum(err_mat, dims=2);

plot!(1:num_it, best_run, fillrange=worst_run, 
    alpha=0.1,
    fillcolor=:purple,
    label="Best-worst interval",
    lineaplha=0,
    );

xlabel!(L"Number of iterations, $m$")
ylabel!(L"$||\widehat{f}_m - f(A)b || / ||f(A)b||$")
title!("Convergence of sFOM on the Wathen Matrix")

display(p)
savefig(p, "~/Documents/Julia/sketched_krylov/figs/sFOMConv.pdf")


