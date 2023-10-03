clc; clear all; close all
wathen_size = 5;

A = gallery('wathen', wathen_size, wathen_size);
A = -A; % make sure exp(A) does not explode

[rows, cols] = size(A);
N = rows;

b = rand(N,1);
b = b/norm(b);

f = @(x) expm(x);

num_it = round(N/2); % must be less than or half for sketch param, can be changed.
ex = expm(A)*b;

num_runs = 1;

err_mat = zeros(num_it, num_runs);

for i = 1:num_runs
    err = sketched_FOM(A, b, f, num_it, ex);
    err_mat(:,i) = err;
end

loglog(1:num_it, err_mat, 'k--o', 'MarkerSize', 10)
grid on 
xlabel("Number of iterations")
ylabel("Absolute error")
title("Error in sFOM")