% addpath('..');
% 
% % Dimension of problem.
% p = 500;
% 
% % Sparse precision matrix.
% A = sprandn(p, p, 1/p);
% A = A'*A;
% X_true = A + 0.1*speye(p); % The precision matrix.
% % G = graph(X_true); % The graph.
% 
% 
% % Covariance matrix and max eigenvalue for subgaussian tail.
% W = inv(full(X_true));
% 
% 
% % Generate data
% 
% nn = [5, 10, 50, 100, 500];
% lam = 10*sqrt(log(p)./nn);
% 
% err = zeros(1, length(nn));
% err_fix = zeros(1, length(nn));
% for i=1:length(nn)
%     data = mvnrnd(zeros(1,p), W, n);
%     
%     S = cov(data);
%     
%     [X, W0] = dpglasso(S, lam(i), 100*p, 1e-4);
%     err(i) = norm(X-X_true, 'Fro');
%     lam(i)
%     [X, W0] = dpglasso(S, 5, 100*p, 1e-4);
%     err_fix(i) = norm(X-X_true, 'Fro');
%     i
% end

figure(1);
loglog(nn, err, 'b-x', 'linewidth', 2);
hold on
loglog(nn, err_fix, 'r-+', 'linewidth', 2);
grid();
legend('$\lambda = 10\sqrt{\log(p)/n}$', '$\lambda = 5$', 'interpreter', 'latex');
xlabel('Sample size $n$', 'interpreter', 'latex');
ylabel('Error $\|\Theta - \Theta_*\|_F$', 'interpreter', 'latex');
title('Error of the graphical lasso for different $\lambda$', 'interpreter', 'latex');
hold off

