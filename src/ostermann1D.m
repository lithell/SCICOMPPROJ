clc; clear all; close all;
%{ 
Run this file to test sFOM on the explicit EUler exponential integrator 
for the Ostermann problem. Homogeneous Dirichlet BCs, and no source term.
%}


% define problem 
N = 100; %space
M = 200; %time

x0 = 0; xN = 1;
h = (xN-x0)/N;

t0 = 0; tM = 0.1;
k = (tM - t0)/M;

% initial cond
u0 = sin(4*pi*(x0+h:h:xN-h));

solmat = zeros(M+1, N-1);
solmat(1,:) = u0;

% disc mat
e = ones(N-1,1);
T = 1/h^2*spdiags([e -2*e e], -1:1, N-1, N-1);


% RHS
G = @(U) T*U + 1./(1+U.^2);

% Jacobian (dunno if this is correct)
J = @(U) T + spdiags(-2*U./(1+U.^2).^2, 0, N-1, N-1);

% phi-func
phi = @(X) X\(expm(X)-speye(size(X)));

% set params for sFOM
max_it = min(floor(0.75*N/2), 100);
trunc_len = 4;
mgs = true;
tol = 10^-11;
verbose = false;

% Don't evaluate error against exact sol in every it 
ex = false;

for i = 1:M

    % forward Euler exponential integrator
    jac = J(solmat(i,:)');
    [~, ~, matfunceval] = sFOM(k*jac, G(solmat(i,:)'), phi, max_it, trunc_len, mgs, ex, tol, verbose);
    solmat(i+1, :) = solmat(i,:) + k*matfunceval';

end


solmat = [zeros(M+1,1), solmat, zeros(M+1,1)];

xvec = x0:h:xN; tvec = t0:k:tM;

[X,T] = meshgrid(xvec, tvec);

figure
surf(X,T, solmat)
xlabel('x')
ylabel('t')
title('Solution to the Ostermann problem')
shading interp


