function [X, W, fvals, dvals] = glasso_admm(S, r, mu, maxiter, tol, ret_obj)
% glasso_admm Solve the graphical lasso problem using ADMM.
%
% S is the p-by-p sample covariance matrix.
% r,mu > 0 are the regularization and penality parameters.
% Returns precision matrix X and covariance matrix W.
% Returns primal and dual objective function values if ret_obj is true.
% 
% Author: Terrence Alsup
% Date: May 18, 2020
% File: glasso_admm.m

% By default do not return objective function values.
if nargin < 6
    ret_obj = false;
end

% Get the dimension of the problem.
p = size(S, 1);

% Initialization.
X = eye(p); % Precision matrix.
Z = eye(p); % Z variable from ADMM
Y = eye(p); % Dual variable

% Return the objective function values after each iteration.
if ret_obj
    % Define the primal objective function.
    obj = @(X) -log(det(X)) + trace(X*S) + r*sum(abs(X), 'all');    
    dobj = @(Y) log(det(S + Y)) + p;
    
    % Initial point is guaranteed to be postive definite (i.e. feasible).
    fvals(1) = obj(X);
    dvals(1) = dobj(Y);
    fprev = fvals(1);
else
    Xprev = X;
end

% Set the initial difference to ensure at least 1 iteration.
rel_diff = 2*tol + 1;

k = 0; % Track the iteration.
while k < maxiter && rel_diff > tol
    
    % Update precision matrix X.
    [V, D] = eig(Z - Y/mu - S/mu);
    d = diag(D);
    D = diag(0.5*(d + sqrt(d.^2 + 4/mu)));
    X = V*D*V';
    
    % Update Z variable with soft thresholding.
    Z = sign(X + Y/mu).*(max(abs(X + Y/mu) - r/mu, 0));
   
    % Update dual variable X.
    Y = Y + mu*(X - Z);
 
    % Increment the iteration.
    k = k + 1;
    
    % Compute the relative difference for the stopping criteria after
    % every iteration.
    if ret_obj
    	% Use relative change in objective function value.
        fvals(k + 1) = obj(X);
        dvals(k + 1) = dobj(Y);
        rel_diff = abs(fprev - fvals(k+1))/abs(fprev);
        fprev = fvals(k + 1);
    else
        % Use relative change in precision matrix.
        rel_diff = norm(X - Xprev, 'Fro')/norm(Xprev, 'Fro');
        Xprev = X;
    end      
end

% Recover the covariance matrix from the dual variable.
W = S + Y;

end

