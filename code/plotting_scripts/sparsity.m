% Spartisty of solution from 3 methods for different values of lambda
addpath('..');

% 0.1 to 10
lam = logspace(-4, 4, 20);

n = 10;
p = 20;
data = randn(n,p);
S = cov(data);
mu = 1.0; % admm

glasso_nnz = zeros(2, length(lam));
dpglasso_nnz = zeros(2, length(lam));
admm_nnz = zeros(2, length(lam));

trials = 25;
tol = 1e-9;
maxit = 100*p;
for i=1:length(lam)
    [X, W, fvals, dvals, extvals] = glasso(S, lam(i), maxit, tol, true);
    glasso_nnz(1,i) = nnz(abs(X) > 1e-4);
    glasso_nnz(2,i) = nnz(abs(X) > 1e-8);
    
    [X, W, fvals, dvals, extvals] = dpglasso(S, lam(i), maxit, tol, true);
    dpglasso_nnz(1,i) = nnz(abs(X) > 1e-4);
    dpglasso_nnz(2,i) = nnz(abs(X) > 1e-8);
    
    [X, W, fvals, dvals, extvals] = glasso_admm(S, lam(i), mu, maxit, tol, true);
    admm_nnz(1,i) = nnz(abs(X) > 1e-4);
    admm_nnz(2,i) = nnz(abs(X) > 1e-8);
end

figure(1);
subplot(1,2,1);
loglog(lam, glasso_nnz(1,:), 'b-x', 'linewidth', 1.5);
hold on
loglog(lam, dpglasso_nnz(1,:), 'r-+', 'linewidth', 1.5);
loglog(lam, admm_nnz(1,:), 'm-o', 'linewidth', 1.5);
grid();
legend('Glasso', 'DP-Glasso', 'ADMM');
xlabel('Regularization parameter $\lambda$', 'interpreter', 'latex');
ylabel('nnz($\Theta \ge 10^{-4}$)', 'interpreter', 'latex');
title('Sparsity of the graphical lasso vs. $\lambda$', 'interpreter', 'latex');
hold off

subplot(1,2,2);
loglog(lam, glasso_nnz(2,:), 'b-x', 'linewidth', 1.5);
hold on
loglog(lam, dpglasso_nnz(2,:), 'r-+', 'linewidth', 1.5);
loglog(lam, admm_nnz(2,:), 'm-o', 'linewidth', 1.5);
grid();
legend('Glasso', 'DP-Glasso', 'ADMM');
xlabel('Regularization parameter $\lambda$', 'interpreter', 'latex');
ylabel('nnz($\Theta \ge 10^{-8}$)', 'interpreter', 'latex');
title('Sparsity of the graphical lasso vs. $\lambda$', 'interpreter', 'latex');
hold off
