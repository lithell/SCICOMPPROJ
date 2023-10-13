clc; clear all; close all
%{
Run this file to test sFOM on the explicit Euler exponential integrator for
the 1D heat eq. Homogeneous Dirichlet BCs, and no source term.
%}

% define problem 
N = 1000; %space
M = 500; %time

x0 = 0; xN = 1;
h = (xN-x0)/N;

t0 = 0; tM = 0.02;
k = (tM - t0)/M;

% initial cond
u0 = sin(4*pi*(x0+h:h:xN-h));

solmat = zeros(M+1, N-1);
solmat(1,:) = u0;

% disc mat
e = ones(N-1,1);
T = 1/h^2*spdiags([e -2*e e], -1:1, N-1, N-1);

% phi-func
phi = @(X) X\(expm(X)-speye(size(X)));

% set params for sFOM
max_it = floor(0.75*N/2);
trunc_len = 4;
mgs = true;
tol = 10^-9;
verbose = false;

% hack
ex_mat = phi(k*T);

final_it = 0;
subplot(1,2,1)
for i = 1:M

    % get exact for presentation of method
    ex = ex_mat*T*solmat(i,:)';

    % forward Euler exponential integrator

    % this is technically not needed and could be precomputed, since the evaluation
    % does not change from iter to iter, for this simple problem 
    [err, final_it, matfunceval] = sFOM(k*T, T*solmat(i,:)', phi, max_it, trunc_len, mgs, ex, tol, verbose);
    solmat(i+1, :) = solmat(i,:) + k*matfunceval';

    semilogy(1:final_it, err, '--', 'LineWidth', 0.5, 'Color', '#0072BD')
    hold on
end

ylabel('Absolute Error')
xlabel('Iteration in sFOM')
title(strcat('Convergence of sFOM for \Delta t=', num2str(k), ', Arnoldi trunc len=', num2str(trunc_len) ))
grid on 

solmat = [zeros(M+1,1), solmat, zeros(M+1,1)];

xvec = x0:h:xN; tvec = t0:k:tM;

[X,T] = meshgrid(xvec, tvec);

subplot(1,2,2)
surf(X,T, solmat)
xlabel('x')
ylabel('t')
shading interp

