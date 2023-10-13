clc; clear all; close all

% Define problem 
wathen_size = 15;

A = gallery('wathen', wathen_size, wathen_size);
A = -A; % make sure exp(A) does not explode

[rows, cols] = size(A);
N = rows;
b = rand(N,1);

% Function in f(A)b
f = @(x) expm(full(x));

num_it = 150; % arbitrary, must be less than or half of N for sketch param

% Arnoldi truncation length
trunc_len = 12;

% Modified or classical Gram-Schmidt orth
mgs = true;

% Exit tol (Note! Will not be same as actual error!)
tol = 10^-10;

% Exact sol
disp(strcat("Computing Exact Solution, N = ", num2str(N)))
ex = f(A)*b;

% Get errors
[err, final_it, appr] = sFOM(A, b, f, num_it, trunc_len, mgs, ex, tol);

semilogy(1:final_it, err, '-o', 'MarkerSize', 5, 'LineWidth', 1.2, 'Color', "#D95319")
grid on 
xlabel("Number of iterations")
ylabel("Absolute error")
title("Error in sFOM")
legend('sFOM')