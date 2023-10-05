function [err] = sketched_FOM_v2(A, b, f, num_it, trunc_len, mgs, ex)

    % block truncated rational Arnoldi with randomized reduction
    % (A,B) - matrix pair
    % v     - block vector
    % xi    - shift parameters (can be infinite)
    % t     - truncation parameter (t=2 will give Lanczos for Hermitian AB)
    % hS    - handle to left-side basis reduction
    % 
    % SV    - left-reduced Krylov basis matrix
    % SAV   - left-reduced A times basis matrix
    % SBV   - left-reduced B times basis matrix
    % Vfull - also return full unreduced, non-orthogonal Krylov basis (optional)


    % initializations
    N = size(A,1);
    err = zeros(num_it,1);
    
    % Set up the sketching
    hS = setup_sketching_handle(N,2*num_it); % s = 2*m_max

    % Allocate for truncated Krylov basis
    V = zeros(N, trunc_len); % truncated orthonormal basis (latest trunc_len vectors)
    
    % normalize v
    [v,~] = qr(b,0);
    
    % add first vector to Krylov basis
    V = [ V(:,2:end) , v ];
    
    % Sketch v and add to sketched basis 
    SV = hS(v); s = size(SV,1);
    SV = [ SV, zeros(s,num_it) ]; %check this
    
    % Allocate for sketched AV
    SAV = zeros(s,num_it+1); 
    
    % Optionally output full basis
    Vfull = zeros(N,num_it+1);
    Vfull(:,1) = v;

    
    % Do sFOM iters
    for m = 1:num_it
    
        % Extract latest Arnoldi vector
        w = V(:,end);
    
        Aw = A*w; 
    
        % compute these retrospectively (?)
        % Sketch Aw, and save 
        SAV(:,m) = hS(Aw); 

        % Update w
        w = Aw;
    
        % Orthogonalize
        if mgs % modified gram schmidt
            for i = 1:t
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
        [SV(:,1:m), SAV(:,1:m), Rw] = whiten_basis(SV(:,1:m), SAV(:,1:m));

        % Compute sFOM approx
        SVm = SV(:,1:m);
        M = SVm'*SVm;
    
        coeffs = M\(f( (SVm'*SAV(:,1:m))/M )*(SVm'*hS(b)));
        appr = Vfull(:,1:m)*(Rw\coeffs)

        % Get error
        err(m) = norm(appr - ex);
    
    end
    
    % final block column of SAV (don't know if this is needed?)
    %SAV(:,1+j*b:(j+1)*b) = hS(A*v); 



end