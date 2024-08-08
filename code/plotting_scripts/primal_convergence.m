% Accuracy of 3 methods for different values of lambda
addpath('..');

% 4 values
lam = [0.1, 0.5, 1, 2];

n = 10;
p = 20;
data = randn(n,p);
S = cov(data);
mu = 1.0; % admm

maxit = 100*p;
tol = 1e-12;

figure(1);
subplot(2, 2, 1);

[X, W, fvals1, dvals1, extvals] = glasso(S, lam(1), maxit, tol, true);
[X, W, fvals2, dvals2, extvals] = dpglasso(S, lam(1), maxit, tol, true);
[X, W, fvals3, dvals3, extvals] = glasso_admm(S, lam(1), mu, maxit, tol, true);
its1 = length(fvals1) - 1;
its2 = length(fvals2) - 1;
its3 = length(fvals3) - 1;
semilogy(0:its1-1, abs(fvals1(1:its1) - fvals1(its1+1)), 'b-x', 'linewidth', 1.5);
hold on
semilogy(0:its2-1, abs(fvals2(1:its2) - fvals2(its2+1)), 'r-+', 'linewidth', 1.5);
semilogy(0:its3-1, abs(fvals3(1:its3) - fvals3(its3+1)), 'm-o', 'linewidth', 1.5);
grid();
legend('Glasso', 'DP-Glasso', 'ADMM');
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Error $|f(\Theta^{(k)}) - f(\Theta^*)|$', 'interpreter', 'latex');
title('Convergence to the primal optimal value $\lambda = 0.1$', 'interpreter', 'latex');
hold off

subplot(2,2,2);

[X, W, fvals1, dvals1, extvals] = glasso(S, lam(2), maxit, tol, true);
[X, W, fvals2, dvals2, extvals] = dpglasso(S, lam(2), maxit, tol, true);
[X, W, fvals3, dvals3, extvals] = glasso_admm(S, lam(2), mu,maxit, tol, true);
its1 = length(fvals1) - 1;
its2 = length(fvals2) - 1;
its3 = length(fvals3) - 1;
semilogy(0:its1-1, abs(fvals1(1:its1) - fvals1(its1+1)), 'b-x', 'linewidth', 1.5);
hold on
semilogy(0:its2-1, abs(fvals2(1:its2) - fvals2(its2+1)), 'r-+', 'linewidth', 1.5);
semilogy(0:its3-1, abs(fvals3(1:its3) - fvals3(its3+1)), 'm-o', 'linewidth', 1.5);
grid();
legend('Glasso', 'DP-Glasso', 'ADMM');
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Error $|f(\Theta^{(k)}) - f(\Theta^*)|$', 'interpreter', 'latex');
title('Convergence to the primal optimal value $\lambda = 0.5$', 'interpreter', 'latex');
hold off

subplot(2,2,3);

[X, W, fvals1, dvals1, extvals] = glasso(S, lam(3), maxit, tol, true);
[X, W, fvals2, dvals2, extvals] = dpglasso(S, lam(3), maxit, tol, true);
[X, W, fvals3, dvals3, extvals] = glasso_admm(S, lam(3), mu,maxit, tol, true);
its1 = length(fvals1) - 1;
its2 = length(fvals2) - 1;
its3 = length(fvals3) - 1;
semilogy(0:its1-1, abs(fvals1(1:its1) - fvals1(its1+1)), 'b-x', 'linewidth', 1.5);
hold on
semilogy(0:its2-1, abs(fvals2(1:its2) - fvals2(its2+1)), 'r-+', 'linewidth', 1.5);
semilogy(0:its3-1, abs(fvals3(1:its3) - fvals3(its3+1)), 'm-o', 'linewidth', 1.5);
grid();
legend('Glasso', 'DP-Glasso', 'ADMM');
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Error $|f(\Theta^{(k)}) - f(\Theta^*)|$', 'interpreter', 'latex');
title('Convergence to the primal optimal value $\lambda = 1$', 'interpreter', 'latex');
hold off

subplot(2,2,4);

[X, W, fvals1, dvals1, extvals] = glasso(S, lam(4), maxit, tol, true);
[X, W, fvals2, dvals2, extvals] = dpglasso(S, lam(4), maxit, tol, true);
[X, W, fvals3, dvals3, extvals] = glasso_admm(S, lam(4), mu,maxit, tol, true);
its1 = length(fvals1) - 1;
its2 = length(fvals2) - 1;
its3 = length(fvals3) - 1;
semilogy(0:its1-1, abs(fvals1(1:its1) - fvals1(its1+1)), 'b-x', 'linewidth', 1.5);
hold on
semilogy(0:its2-1, abs(fvals2(1:its2) - fvals2(its2+1)), 'r-+', 'linewidth', 1.5);
semilogy(0:its3-1, abs(fvals3(1:its3) - fvals3(its3+1)), 'm-o', 'linewidth', 1.5);
grid();
legend('Glasso', 'DP-Glasso', 'ADMM');
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Error $|f(\Theta^{(k)}) - f(\Theta^*)|$', 'interpreter', 'latex');
title('Convergence to the primal optimal value $\lambda = 2$', 'interpreter', 'latex');
hold off