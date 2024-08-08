function [x, fvals] = QP_box(Q, b, l, u, x0, maxiter, tol, ret_obj)
% QP_box Solve the QP with box constraints using coordinate descent.
%
% min 0.5*x'Qx + b'x
% s.t. l <= x <= u
%
% where Q is a positive definite p-by-p matrix.
% x0,b,l,u are column vectors of dimension p-by-1.
% Returns the optimals solution x.
% Returns the function values fvals if ret_obj is true.
% 
% Author: Terrence Alsup
% Date: May 18, 2020
% File: QP_box.m


% By default do not return the objective function values.
if nargin < 8
    ret_obj = false;
end

% Get the dimension of the problem.
p = size(x0, 1);

% Set the starting value.
x = x0;

% Return the objective function values after each cycle.
if ret_obj
    obj = @(x) 0.5*x'*Q*x + b'*x;
    fvals(1) = obj(x);
    fprev = fvals(1);
else
    xprev = x;
end

% Set the initial difference to ensure at least 1 iteration.
rel_diff = 2*tol + 1;

k = 0; % Track the iteration.
while k < maxiter && rel_diff > tol

    % Get the coordinate to update.
    i = mod(k, p) + 1;
	% Perform update on coordinate i.
    temp = (Q(i,i)*x(i) - b(i) - Q(i,:)*x)/Q(i,i);
    % Threshold to meet box constraints.
    x(i) = max(min(temp, u(i)), l(i));

    % Increment the iteration
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

