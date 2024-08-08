% Verify the QP with box constraints using CVX.


cvx_begin
    variable x(p)
    minimize (0.5*x'*Q*x + b'*x)
    subject to
        l <= x <= u
cvx_end

x;