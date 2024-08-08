% Plot |X^-1 - W|_F for glasso, dpglasso, admm
addpath('..');

% Random problem.
p = 50;
X = randn(5, p);
S = cov(X);
r = 1.0;
mu = 1.0;
[X1, W1, fvals1, dvals1, extvals1] = glasso(S, r, 50*p, 1e-9, true);
[X2, W2, fvals2, dvals2, extvals2] = dpglasso(S, r, 50*p, 1e-9, true);
[X3, W3, fvals3, dvals3, extvals3] = glasso_admm(S, r, mu, 50*p, 1e-9, true);

figure(1);
semilogy(1:length(extvals1)-1, extvals1(2:end), 'b-x', 'linewidth', 1.5);
hold on
semilogy(1:length(extvals2)-1, extvals2(2:end), 'r-+', 'linewidth', 1.5);
semilogy(1:length(extvals3)-1, extvals3(2:end), 'm-o', 'linewidth', 1.5);
grid();
legend('Glasso', 'DP-Glasso', 'ADMM', 'interpreter', 'latex');
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('$\|\Sigma \Theta - I\|_F$', 'interpreter', 'latex');
title('Error of $\Sigma\Theta - I$', 'interpreter', 'latex');
hold off