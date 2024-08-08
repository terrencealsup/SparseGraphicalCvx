function [X, W, fvals, dvals] = pglasso(S, r, maxiter, tol, ret_obj)
% pglasso Solve the graphical lasso problem using the P-Glasso algorithm.
%   Detailed explanation goes here

% By default do not return objective function values.
if nargin < 5
    ret_obj = false;
end

% Get the dimension of the problem.
p = size(S, 1);

% Initialize the precision and covariance matrices.
W = diag(diag(S) + r);
X = spdiags(1./diag(W), 0, p, p);

% Dual feasible point.
%W = S + r*eye(p);
%X = inv(W);


% Return the objective function values after each cycle.
if ret_obj
    % Define the primal objective function.
    obj = @(X) -log(det(X)) + trace(X*S) + r*sum(abs(X), 'all');
    
    dobj = @(W) log(det(W)) + p;
    
    % Initial point is guaranteed to be postive definite (i.e. feasible).
    fvals(1) = obj(X);
    dvals(1) = dobj(W);
    fprev = fvals(1);
else
    Xprev = X;
end

% Set the initial difference to ensure at least 1 iteration.
rel_diff = 2*tol + 1;

k = 0; % Track the iteration.
while k < maxiter && rel_diff > tol

    % The coordinate to update.
    i = mod(k, p) + 1;
 	idx = cat(2, 1:i-1, i+1:p);
    
	W11 = W(idx, idx);
	w12 = W(idx, i);
	w21 = W(i, idx);
    w22 = W(i,i);
    
	s12 = S(idx, i);

    X11_inv = W11 - w12*w21/w22;
	x12 = X(idx, i);
    X11 = X(idx,idx);

	% Solve for alpha. 
	a = lasso(X11_inv, s12, r, w22*x12, 100*p, 1e-6);
    
	x12 = a/w22;
	x22 = x12'*X11_inv*x12 + 1/w22;
    
	% Update X and W.
	w12 = -X11_inv*x12*w22;
    W11 = X11_inv + w12*w12'/w22;
    
    W(idx, idx) = W11;
	W(idx, i) = w12;
	W(i, idx) = w12';
    
	X(idx, i) = x12;
	X(i, idx) = x12';
	X(i,i) = x22;

    X11_inv = W11 - w12*w21/w22;
    X(idx,idx) = inv(X11_inv);
   

    % Increment the iteration.
    k = k + 1;
    
    % Compute the relative difference for the stopping criteria after
    % every cycle over all coordinates.
    if ret_obj
    	% Use relative change in objective function value.
        fvals(k + 1) = obj(X);
        dvals(k + 1) = dobj(W);
        if i == p
            rel_diff = (fprev - fvals(k+1))/abs(fprev);
            fprev = fvals(k + 1);
            norm(X*W - eye(p), 'Fro')
        end
    else
        % Use relative change in precision matrix.
        if i == p
            rel_diff = norm(X - Xprev, 'Fro')/norm(Xprev, 'Fro');
            Xprev = X;
        end
    end     

end

end

