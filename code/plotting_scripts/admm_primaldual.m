% Plot the primal and dual values for DP-Glasso.
addpath('..');

p = 50;
X = randn(5, p);
S = cov(X);
r = 0.5;
mu = 1.0;
[X, W, fvals, dvals, ddvals] = glasso_admm(S, r, mu, p, 1e-9, true);


figure(1);

ax1 = subplot(1, 2, 1);
plot(0:length(fvals)-1, fvals, 'b-*', 'linewidth', 1.5);
hold on
plot(0:length(dvals)-1, dvals, 'r-+', 'linewidth', 1.5);
hold off
grid();
legend('Primal', 'Dual');
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Objective function values', 'interpreter', 'latex');
title('ADMM: Dual objective function at $Y$', 'interpreter', 'latex');

ax2 = subplot(1, 2, 2);
plot(0:length(fvals)-1, fvals, 'b-*', 'linewidth', 1.5);
hold on
plot(0:length(ddvals)-1, ddvals, 'r-+', 'linewidth', 1.5);
hold off
grid();
legend('Primal', 'Dual');
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Objective function values', 'interpreter', 'latex');
title('ADMM: Dual objective function at $\Theta^{-1} - S$', 'interpreter', 'latex');

linkaxes([ax1 ax2],'xy');