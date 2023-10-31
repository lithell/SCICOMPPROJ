function setupSketchingHandle(N, sketch_param)

    # Set up randomly subsampled dct sketching

    # construct random E
    rows = 1:N;
    cols = 1:N;
    vals = 2*round.(rand(N));
    E = sparse(rows, cols, vals);

    # construct random D
    D = sparse(I, N, N);
    D = D[rand(1:N, sketch_param), :];

    # Return sketching function
    return x -> D*dct(E*x,1)/sqrt(sketch_param/N);

end
