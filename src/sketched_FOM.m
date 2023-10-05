function [err] = sketched_FOM(A, b, f, num_it, ex)

    %{
    Function for performing the sketched FOM iterations.
    
    INPUT: matrix A, vector b, function handle f, 
           number of iterations num_it, exact solution ex
    OUTPUT: num_it x 1 vector of errors corresponding to the 
            approximations of f(A)b.
    %}

    % initializations
    N = size(A,1);
    
    % get Arnolid basis
    k = 4; % how many blocks to orthogonalise against
    hS = setup_sketching_handle(N,2*num_it); % s = 2*m_max
    [SV,SAV,~,Vtrunc] = bta(A,speye(N),b,inf(1,num_it),k,hS); % number of mat-vec products = num_it
    
    % whitening the basis
    [SV, SAV, Rw] = whiten_basis(SV, SAV);

    
    % sFOM   
    err = zeros(num_it,1);

    for m = 1:num_it, m
        SVm = SV(:,1:m);
        M = SVm'*SVm;
    
        coeffs = M\(f( (SVm'*SAV(:,1:m))/M )*(SVm'*hS(b)));
        appr = Vtrunc(:,1:m)*(Rw(1:m,1:m)\coeffs);
        err(m) = norm(appr - ex);

    end
end