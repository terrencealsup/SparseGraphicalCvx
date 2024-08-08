function [X, W, fvals, dvals] = glasso(S, r, maxiter, tol, ret_obj)
% glasso Solve the graphical lasso problem using the glasso algorithm.
%
% S is the p-by-p sample covariance matrix.
% r > 0 is the regularization parameter.
% Returns precision matrix X and covariance matrix W.
% Returns primal and dual objective function values if ret_obj is true.
% 
% Author: Terrence Alsup
% Date: May 18, 2020
% File: glasso.m


% By default do not return objective function values.
if nargin < 5
    ret_obj = false;
end

% Get the dimension of the problem.
p = size(S, 1);

% Initialize the covariance and precision matrices.
W = S + r*eye(p);
X = speye(p); % Do not need to start with X^-1 = W.

% Return the objective function values after each iteration.
if ret_obj
    % Define the primal objective function.
    obj = @(X) -log(det(X)) + trace(X*S) + r*sum(abs(X), 'all');   
    dobj = @(W) log(det(W)) + p;
    
    % Initial point is guaranteed to be postive definite (i.e. feasible).
    fvals(1) = obj(X);
    dvals(1) = dobj(W);
    fprev = fvals(1);
else
    % Track the covariance matrix.
    Wprev = W;
end

% Set the initial difference to ensure at least 1 iteration.
rel_diff = 2*tol + 1;

k = 0; % Track the iteration.
while k < maxiter && rel_diff > tol
    
    % Get the coordinate to update.
    i = mod(k, p) + 1;
    
    % Get the indices for the block submatrix.
    idx = cat(2, 1:i-1, i+1:p);

    W11 = W(idx, idx);
    w12 = W(idx, i);
    w22 = W(i, i);    
    s12 = S(idx, i);    
    x12 = X(idx, i);
    x22 = X(i, i);
    
    % Solve the lasso problem with coordinate descent.
    % Starting point is previous optimal value of b.
    b = lasso(W11, s12, r, x12/x22, 100*p, 1e-6);

    % Update the covariance matrix.
    w12 = -W11*b;
    W(idx, i) = w12;
    W(i, idx) = w12';
    
    % Update the precision matrix.
    x22 = 1/(w22 + b'*w12);
    x12 = x22*b;
    X(i, i) = x22;
    X(idx, i) = x12;
    X(i, idx) = x12';

    
    % Increment the iteration.
    k = k + 1;
    
    % Compute the relative difference for the stopping criteria after
    % every cycle over all coordinates.
    if ret_obj
    	% Use relative change in objective function value.
        % Note that we use inv(W) instead of X because WX != I.
        fvals(k + 1) = obj(X);
        dvals(k + 1) = dobj(W);
        if i == p
            rel_diff = (fprev - fvals(k+1))/abs(fprev);
            fprev = fvals(k + 1);
        end
    else
        if i == p
            % Use relative change in covariance matrix.
            rel_diff = norm(W - Wprev, 'Fro')/norm(Wprev, 'Fro');
            Wprev = W;
        end
    end    
end
end

