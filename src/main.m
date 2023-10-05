clc; clear all; close all
wathen_size = 9;

A = gallery('wathen', wathen_size, wathen_size);
A = -A; % make sure exp(A) does not explode

[rows, cols] = size(A);
N = rows;

b = rand(N,1);
b = b/norm(b);

f = @(x) expm(x);

num_it = 80;
%num_it = round(N/2); % must be less than or half for sketch param, can be changed.

trunc_len = 4;
mgs = false;
ex = f(A)*b;

err = sketched_FOM_v2(A, b, f, num_it, trunc_len, mgs, ex);

loglog(1:num_it, err, 'k--o', 'MarkerSize', 5)
grid on 
xlabel("Number of iterations")
ylabel("Absolute error")
title("Error in sFOM")