% Verify the lasso solution using CVX.


cvx_begin
    variable x(p)
    minimize (0.5*x'*A*x + b'*x + r*norm(x, 1))
cvx_end

x;