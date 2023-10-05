clc; clear all; close all

wathen_size = 25;

A = gallery('wathen', wathen_size, wathen_size);
A = -A; % make sure exp(A) does not explode

[rows, cols] = size(A);
N = rows;

b = rand(N,1);

% Function in f(A)b
f = @(x) expm(full(x));

num_it = 175; % must be less than or half of N for sketch param, can be changed.

% Arnoldi truncation length
trunc_len = 12;

% Modified or classical Gram-Schmidt orth
mgs = true;

% Exact sol
ex = f(A)*b;

% Get errors
err = sFOM(A, b, f, num_it, trunc_len, mgs, ex);

loglog(1:num_it, err, 'k-o', 'MarkerSize', 5, 'LineWidth', 1.2)
grid on 
xlabel("Number of iterations")
ylabel("Absolute error")
title("Error in sFOM")
legend('sFOM')