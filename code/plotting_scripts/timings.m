% Time the 3 methods for different values of lambda
addpath('..');

% 0.1 to 10
lam = logspace(-1, 1, 10);

n = 20;
p = 50;
data = randn(n,p);
S = cov(data);
mu = 1.0; % admm

glasso_its = zeros(1, length(lam));
glasso_time = zeros(1, length(lam));
dpglasso_its = zeros(1, length(lam));
dpglasso_time = zeros(1, length(lam));
admm_its = zeros(1, length(lam));
admm_time = zeros(1, length(lam));

trials = 25;
tol = 1e-9;
maxit = 100*p;
for i=1:length(lam)
    tic;
    for j=1:trials
        [X, W, fvals, dvals, extvals] = glasso(S, lam(i), maxit, tol, true);
        glasso_its(i) = glasso_its(i) + length(fvals) - 1;
    end
    glasso_time(i) = toc/trials;
    
    tic;
    for j=1:trials
        [X, W, fvals, dvals, extvals] = dpglasso(S, lam(i), maxit, tol, true);
        dpglasso_its(i) = dpglasso_its(i) + length(fvals) - 1;
    end
    dpglasso_time(i) = toc/trials;
    
    tic;
    for j=1:trials
        [X, W, fvals, dvals, extvals] = glasso_admm(S, lam(i), mu, maxit, tol, true);
        admm_its(i) = admm_its(i) + length(fvals) - 1;
    end
    admm_time(i) = toc/trials;
    
    i
end

figure(1);
subplot(1,2,1);
semilogx(lam, glasso_its, 'b-x', 'linewidth', 1.5);
hold on
semilogx(lam, dpglasso_its, 'r-+', 'linewidth', 1.5);
semilogx(lam, admm_its, 'm-o', 'linewidth', 1.5);
grid();
legend('Glasso', 'DP-Glasso', 'ADMM');
xlabel('Regularization parameter $\lambda$', 'interpreter', 'latex');
ylabel('Iterations to convergence', 'interpreter', 'latex');
title('Average number of iterations required for convergence vs. $\lambda$', 'interpreter', 'latex');
hold off


subplot(1,2,2);
semilogx(lam, glasso_time, 'b-x', 'linewidth', 1.5);
hold on
semilogx(lam, dpglasso_time, 'r-+', 'linewidth', 1.5);
semilogx(lam, admm_time, 'm-o', 'linewidth', 1.5);
grid();
legend('Glasso', 'DP-Glasso', 'ADMM');
xlabel('Regularization parameter $\lambda$', 'interpreter', 'latex');
ylabel('Runtime [s]', 'interpreter', 'latex');
title('Runtime required for convergence vs. $\lambda$', 'interpreter', 'latex');
hold off