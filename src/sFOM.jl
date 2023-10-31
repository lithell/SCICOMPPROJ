"""
    sFOM(A, b, f, num_it, trunc_len, mgs, iter_diff_tol, sketch)

Compute the sFOM approximation `fₘ≈f(A)b`.

# Arguments
- `A::Matrix`
- `b::Vector`
- `f::Function`
- `num_it::Int` Max number of iterations
- `trunc_len::Int` Arnoldi truncation length
- `mgs::Bool` Use MGS orthogonalization, otherwise use GS
- `iter_diff_tol::Float` Iteration-difference tolerance
- `sketch::Function` Sketching function in sFOM
"""
function sFOM(A, b, f, num_it, trunc_len, mgs, iter_diff_tol, sketch)

    # Convergence flag
    conv_flag = -1;

    # Initializations
    N = size(A, 1);
    iter_diff = zeros(num_it);

    # Allocate for truncated Krylov basis
    V = zeros(ComplexF64, N, trunc_len);

    # Add first Krylov vector
    v = b/norm(b);
    V = hcat(V[:,2:end], v);

    # Sketch v and add to sketched basis
    SV = sketch(v);
    sketch_size = size(SV,1);
    SV = hcat(SV, zeros(ComplexF64, sketch_size, num_it));

    # Allocate for sketched AV
    SAV = zeros(ComplexF64, sketch_size, num_it+1);

    # Allocate for full Krylov basis
    Vfull = zeros(ComplexF64, N, num_it+1);
    Vfull[:,1] = v;

    # init approx, qm
    approx = zeros(size(b));
    qm = 0;

    # Do sFOM iters
    for m = 1:num_it

        # Compute mat-vec product with latest Arnolid vec
        v = V[:,end];
        Av = A*v;

        # Sketch Av and add to sketched basis 
        SAV[:,m] = sketch(Av);

        # Update v
        v = Av;

        # Orthogonalize
        if mgs
            for i=1:trunc_len
                v = v - V[:,i]*(V[:,i]'*v);
            end
        else
            v = v - V*(V'*v);
        end

        # Normalize 
        v = v/norm(v);

        # Add to truncated basis, and discard vectors below trunc_len
        V = hcat(V[:,2:end], v);

        # Update sketched V
        SV[:,m+1] = sketch(v);

        # Save v in full Krylov basis
        Vfull[:,m+1] = v;

        # Whiten basis
        SVw, SAVw, Rw = whitenBasis(view(SV,:,1:m), view(SAV,:,1:m));

        # Save previous approx
        if m >= 2
            qm_prev = qm;
        end

        # Compute sFOM qₘ
        SVm = SVw;
        M = SVm'*SVm;
        coeffs = M\( f((SVm'*SAVw)/M)*(SVm'*sketch(b)) );
        qm = (Rw\coeffs);

        # Evaluate stopping criterion 
        if m >= 2

            stop_crit = 1 / norm(SV[:,m]);
            stop_crit *= norm(SV[:,1:m]*(qm - vcat(qm_prev, 0)));
            iter_diff[m] = stop_crit;

            if stop_crit < iter_diff_tol

                # Compute full approx
                approx = view(Vfull,:,1:m)*qm;
                iter_diff = iter_diff[1:m];
                conv_flag = 1;

                return approx, conv_flag, iter_diff;
            end

        end

    end

    approx = view(Vfull,:,1:num_it)*qm;
    
    return approx, conv_flag, iter_diff;

end
