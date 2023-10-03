function [err] = sketched_FOM(A, b, f, num_it, ex)

    % initializations
    N = size(A,1); 
    b = b/norm(b);

    
    k = 10; % how many blocks to orthogonalise against
    hS = setup_sketching_handle(N,2*num_it); % s = 2*m_max
    [SV,SAV,~,Vtrunc] = bta(A,speye(N),b,inf(1,num_it),k,hS); % number of mat-vec products = num_it
    
    % whitening the basis
    [SV, SAV, Rw] = whiten_basis(SV, SAV);

    
    % sFOM
    
    Sb = hS(b);

    err = zeros(num_it,1);

    for m = 1:num_it, m
        SVm = SV(:,1:m);
        M = SVm'*SVm;
    
        coeffs = M\(f( (SVm'*SAV(:,1:m))/M )*(SVm'*Sb));
        appr = Vtrunc(:,1:m)*(Rw(1:m,1:m)\coeffs);
        err(m) = norm(appr - ex);
        %f_sk =  Rw(1:m,1:m)\(M\(f((SVm'*SAV(:,1:m))/M)));
        %f_tr = f(Vfull(:,1:m)'*A*Vfull(:,1:m));
        %diff_f(m) = norm(f_sk-f_tr)/norm(f_tr);
    end
end