function [err] = sFOM(A, b, f, num_it, trunc_len, mgs, ex)

    %{
    Function for performing the sketched FOM iterations.
    
    INPUT: matrix A, vector b, function handle f, 
           number of iterations num_it, Arnolid truncation length exact solution ex
    OUTPUT: num_it x 1 vector of errors corresponding to the 
            approximations of f(A)b.

    TODO: Change set number of iterations to exit condition.
    %}

    % Initializations
    N = size(A,1);
    err = zeros(num_it,1);
    
    % Set up the sketching
    hS = setup_sketching_handle(N,2*num_it); % s = 2*m_max

    % Allocate for truncated Krylov basis
    V = zeros(N, trunc_len); % truncated orthonormal basis (latest trunc_len vectors)
    
    % Normalize v
    [v,~] = qr(b,0);
    
    % Add first vector to Krylov basis
    V = [ V(:,2:end) , v ];
    
    % Sketch v and add to sketched basis 
    SV = hS(v); s = size(SV,1);
    SV = [ SV, zeros(s,num_it) ];
    
    % Allocate for sketched AV
    SAV = zeros(s,num_it+1); 
    
    % Allocate for full Krylov basis 
    Vfull = zeros(N,num_it+1);
    Vfull(:,1) = v;

    % Do sFOM iters
    for m = 1:num_it
    
        % Extract latest Arnoldi vector
        w = V(:,end);
    
        Aw = A*w; 

        % Sketch Aw, and save 
        SAV(:,m) = hS(Aw); 

        % Update w
        w = Aw;
    
        % Orthogonalize
        if mgs % modified gram schmidt
            for i = 1:trunc_len
                w = w - V(:,i)*(V(:,i)'*w);
            end
        else % classical gram schmidt
            w = w - V*(V'*w);
        end
    
        % Normalize w
        [v,~] = qr(w,0);
    
        % Add to truncated basis, and discard vectors below truncation length
        V = [ V(:,2:end) , v ];
        
        % Update sketched V
        SV(:,m+1) = hS(v); 
        
        % Save v in full Krylov basis
        Vfull(:,m+1) = v;

        % Whiten basis
        [SVw, SAVw, Rw] = whiten_basis(SV(:,1:m), SAV(:,1:m));

        % Compute sFOM approx
        SVm = SVw;
        M = SVm'*SVm;  
        coeffs = M\(f( (SVm'*SAVw)/M )*(SVm'*hS(b)));
        appr = Vfull(:,1:m)*(Rw\coeffs);

        % Get error
        err(m) = norm(appr - ex);
    
    end

end