% Plot the convergence CD for the QP with box constraints and the lasso.
addpath('..');

% Create a random problem.
p = 50;
A = randn(10, p);
% Enforce the matrix to be positive definite with min eigenvalue 1.
A = A'*A + eye(p);  
b = randn(p, 1);
x0 = randn(p, 1);
maxit = 100000;
tol = 1e-12;


% Box QP.
l = randn(p, 1);
u = l + 1 + 10*abs(randn(p, 1));
[x1, fvals1] = QP_box(A, b, l, u, x0, maxit, tol, true);
numits1 = length(fvals1) - 1;


% Lasso problem
r = 1.0;
[x2, fvals2] = lasso(A, b, r, x0, maxit, tol, true);
numits2 = length(fvals2) - 1;



% Create another random problem, but with larger eigenvalues for A.
p = 50;
A = randn(10, p);
% Enforce the matrix to be positive definite with min eigenvalue 10.
A = A'*A + 10*eye(p);  
b = randn(p, 1);
x0 = randn(p, 1);
maxit = 100000;
tol = 1e-12;


% Box QP.
l = randn(p, 1);
u = l + 1 + 10*abs(randn(p, 1));
[x3, fvals3] = QP_box(A, b, l, u, x0, maxit, tol, true);
numits3 = length(fvals3) - 1;


% Lasso problem
r = 1.0;
[x4, fvals4] = lasso(A, b, r, x0, maxit, tol, true);
numits4 = length(fvals4) - 1;




% Plot the results.
figure(1);

% QP box
subplot(2, 2, 1);
semilogy(1:p:numits1, fvals1(1:p:numits1) - fvals1(numits1 + 1), 'b-o', 'linewidth', 1.5);
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Error $f(x^{(k)}) - f(x^*)$', 'interpreter', 'latex');
title('Convegence of CD for the QP with box constraints', 'interpreter', 'latex');
grid();

subplot(2, 2, 2);
semilogy(1:p:numits2, fvals2(1:p:numits2) - fvals2(numits2 + 1), 'r-s', 'linewidth', 1.5);
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Error $f(x^{(k)}) - f(x^*)$', 'interpreter', 'latex');
title('Convegence of CD for the lasso problem', 'interpreter', 'latex');
grid();


subplot(2, 2, 3);
semilogy(1:p:numits3, fvals3(1:p:numits3) - fvals3(numits3 + 1), 'b-o', 'linewidth', 1.5);
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Error $f(x^{(k)}) - f(x^*)$', 'interpreter', 'latex');
title('Convegence of CD for the QP with box constraints', 'interpreter', 'latex');
grid();

subplot(2, 2, 4);
semilogy(1:p:numits4, fvals4(1:p:numits4) - fvals4(numits4 + 1), 'r-s', 'linewidth', 1.5);
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Error $f(x^{(k)}) - f(x^*)$', 'interpreter', 'latex');
title('Convegence of CD for the lasso problem', 'interpreter', 'latex');
grid();