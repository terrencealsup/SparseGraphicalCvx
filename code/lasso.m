function [x, fvals] = lasso(A, b, r, x0, maxiter, tol, ret_obj)
% lasso Solve the lasso problem using coordinate descent.
%   
%   min 0.5*x'Ax + b'x + r|x|_1]
%
% where A is a positive definite p-by-p matrix.
% x0, b are p-by-1 vectors and r > 0 is the regularization.
% Returns the optimal solution x.
% Returns the function values fvals if ret_obj is true.
%
% Author: Terrence Alsup
% Date: May 18, 2020
% File: lasso.m

% By default do not return the objective function values.
if nargin < 7
    ret_obj = false;
end

% Get the dimension of the problem.
p = size(x0, 1);

% Set the starting value.
x = x0;

% Return the objective function values after each cycle.
if ret_obj
    obj = @(x) 0.5*x'*A*x + b'*x + r*norm(x, 1);
    fvals(1) = obj(x);
    fprev = fvals(1);
else
    xprev = x;
end

% Set the initial difference to ensure at least 1 iteration.
rel_diff = 2*tol + 1;

k = 0; % Track the iteration.
while k < maxiter && rel_diff > tol

    % The coordinate to update.
    i = mod(k, p) + 1;
	% Perform the update to coordinate i.
	temp = (-b(i) - A(i,:)*x + A(i,i)*x(i));
	% Soft thresholding.
	x(i) = sign(temp)*max(abs(temp) - r, 0)/A(i,i);

    % Increment the iteration.
    k = k + 1;
    
    % Compute the relative difference for the stopping criteria after
    % every cycle over all coordinates.
    if ret_obj
    	% Use relative change in objective function value.
        fvals(k + 1) = obj(x);
        if i == p
            rel_diff = (fprev - fvals(k+1))/abs(fprev);
            fprev = fvals(k + 1);
        end
    else
        % Use relative change in solution.
        if i == p
            rel_diff = norm(x - xprev)/norm(xprev);
            xprev = x;
        end
    end    
end
end

