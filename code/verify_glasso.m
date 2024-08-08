cvx_begin sdp

    variable X(p,p) symmetric
    minimize( -log_det(X) + trace(X*S) + r*sum(sum(abs(X))))
    X >= 0

cvx_end

X;