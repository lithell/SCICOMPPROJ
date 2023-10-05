function [SV,SAV,SBV,Vfull] = bta(A,B,v,xi,t,hS,mgs,reo)
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

[N,b] = size(v);

if nargin < 7
    mgs = false;
end
if nargin < 8
    reo = 0;
end
if ishermitian(A) && ishermitian(B) && all(isreal(xi))
    disp('Hermitian problem with real shifts. (LDL flag = 1, only for mylinsolve.m)')
    ldl_flag = 1;
else
    ldl_flag = 0;
end

if norm(B - speye(N),'fro') < eps
    disp('B is identity matrix.')
    identityB = 1;
else
    identityB = 0;
end


V = zeros(N, t*b); % truncated orthonormal basis (last t blocks)

mylinsolve(); % using decomposition, slower but more accurate
%mylinsolve_umfpack(); % fast, less accuracte, used by eigs?
%util_linsolve(); % LU, faster but less accurate

% normalize v
[v,~] = qr(v,0);

% add first vector to Krylov basis
V = [ V(:,b+1:end) , v ];

% Sketch v and add to sketched basis 
SV = hS(v); s = size(SV,1);
SV = [ SV, zeros(s,length(xi)*b) ];

% Allocate for sketched AV & BV
SAV = zeros(s,(length(xi)+1)*b); 
SBV = zeros(s,(length(xi)+1)*b);

% Optionally output full basis
Vfull = [];
if nargout>=4
    Vfull = zeros(N,(length(xi)+1)*b);
    Vfull(:,1:b) = v;
end

% Do Arnolid iters
wb = waitbar(0,'bta running');
for j = 1:length(xi)

    % Extract latest Arnoldi vector (may be block)
    w = V(:,end-b+1:end);

    Aw = A*w; 

    % compute these retrospectively (?)
    % Sketch Aw, and save 
    SAV(:,1+(j-1)*b:(j)*b) = hS(Aw); 

    % Take care of B (not neede for our purposes)
    if identityB
        Bw = w;
        SBV(:,1+(j-1)*b:(j)*b) = SV(:,1+(j-1)*b:(j)*b); 
    else
        Bw = B*w;
        SBV(:,1+(j-1)*b:(j)*b) = hS(Bw); 
    end
    
    % Shifted Arnoldi?
    if isfinite(xi(j))
        w = mylinsolve(A - xi(j)*B, Bw, ldl_flag); % slower
        %w = mylinsolve_umfpack(A - xi(j)*B, Bw); % used by eigs?
        %w = util_linsolve(A - xi(j)*B, Bw); 
    else
        if identityB % Our case!
            w = Aw;
        else
            w = mylinsolve(B, Aw, ldl_flag);
            %w = mylinsolve_umfpack(B, Aw); % used by eigs?
            %w = util_linsolve(B, Aw);
        end
    end

    % Orthogonalize
    if mgs % modified gram schmidt
        for reo = 0:1 % Reorthogonalization
            for i = 1:t
                w = w - V(:,1+(i-1)*b:i*b)*(V(:,1+(i-1)*b:i*b)'*w);
            end
        end
    else % classical gram schmidt
        w = w - V*(V'*w);
    end

    % Normalize w
    [v,~] = qr(w,0);

    % Add to truncated basis, and discard vectors below truncation length
    V = [ V(:,b+1:end) , v ];
    
    % Update sketched V
    SV(:,1+j*b:(j+1)*b) = hS(v); 
    
    % If we want full basis, save v
    if nargout>=4
        Vfull(:,1+j*b:(j+1)*b) = v;
    end
    waitbar(j/length(xi),wb)
end

% final block column of SAV and SBV
SAV(:,1+j*b:(j+1)*b) = hS(A*v); 
if identityB
    SBV(:,1+j*b:(j+1)*b) = SV(:,1+j*b:(j+1)*b);
else
    SBV(:,1+j*b:(j+1)*b) = hS(B*v);
end

close(wb)
end
