% Plot the primal and dual values for the glasso.
addpath('..');

p = 50;
X = 1.2*randn(5, p);
S = cov(X);
r = 0.5;
[X, W, fvals, dvals, ffvals] = glasso(S, r, 20*p, 1e-9, true);


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
title('Glasso: Primal objective function at $\Theta$', 'interpreter', 'latex');

ax2 = subplot(1, 2, 2);
plot(0:length(ffvals)-1, ffvals, 'b-*', 'linewidth', 1.5);
hold on
plot(0:length(dvals)-1, dvals, 'r-+', 'linewidth', 1.5);
hold off
grid();
legend('Primal', 'Dual');
xlabel('Iteration $k$', 'interpreter', 'latex');
ylabel('Objective function values', 'interpreter', 'latex');
title('Glasso: Primal objective function at $\Sigma^{-1}$', 'interpreter', 'latex');

linkaxes([ax1 ax2],'xy');