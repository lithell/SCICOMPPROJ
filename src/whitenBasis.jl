function whitenBasis(SV, SAV)

    Q, Rw = qr(SV);
    Q = Matrix(Q); # Maybe there is a more efficient way than this? But need thin Q explicitly
    SV = Q;
    SAV = SAV/Rw;

    return SV, SAV, Rw

end
