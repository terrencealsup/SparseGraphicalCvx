% p = 50;
% Q = randn(p, p);
% Q = 20*Q'*Q;
% b = randn(p, 1);
% l = randn(p, 1);
% u = l + 50*abs(randn(p, 1));
% x0 = randn(p, 1);
% maxiter = 1000;
% tol = 1e-8;
% 
% 
% [x1, fvals] = QP_box(Q, b, l, u, x0, maxiter, tol, true);
% 
% verify_QP_box;
% 
% norm(x-x1)
% fvals(length(fvals))
% 
% plot(fvals);

p = 50;
X = randn(5, p);
S = cov(X);
r = 0.5;
[X0, W0, fvals0, dvals0] = glasso(S, r, 50*p, 1e-9, true);
[X1, W1, fvals1, dvals1] = dpglasso(S, r, 50*p, 1e-9, true);
%[X2, W2, fvals2, dvals2] = pglasso(S, r, 50*p, 1e-9, true);
%[X3, W3, fvals3, dvals3] = glasso_admm(S, r, 1.0, 100, 1e-9, true);


%verify_glasso

% n = logspace(-15, 2, 50);
% t = zeros(50,1);
% for i=1:50
%     t(i) = nnz(abs(X1) > n(i));
% end
% 
% semilogx(n,t)
%verify_glasso

%norm(X1-X,'Fro')

figure(1)
plot(0:length(fvals0)-1, fvals0, '-*')
hold on
plot(0:length(dvals0)-1, dvals0, '-+')
hold off

figure(2)
plot(0:length(fvals1)-1, fvals1, '-*')
hold on
plot(0:length(dvals1)-1, dvals1, '-+')
hold off

% figure(3)
% plot(0:length(fvals2)-1, fvals2, '-*')
% hold on
% plot(0:length(dvals2)-1, dvals2, '-+')
% hold off
% 
% 
% figure(2)
% spy(abs(X) > 1e-6)


% p = 20;
% X = randn(25, p);
% S = cov(X);
% r = 0.9;
% mu = 1;
% maxiter = 1000;
% 
% Y = glasso_admm(S, r, mu, maxiter, 0);
% verify_glasso;
% 
% norm(X - Y)


% p = 10;
% mu = zeros(p, 1);
% R = randn(p, p);
% sigma = R'*R + eye(p);
% eig(sigma)
% n = 25;
% X = mvnrnd(mu, sigma, n);
% S = cov(X);
% r = 0.1;
% [X1, W, fvals, dvals] = glasso(S, r, 1000, 0, true);
% 
% verify_glasso
% 
% obj = @(Y) -log(det(Y)) + trace(Y*S) + r*sum(abs(Y), 'all');
% dobj = @(Y) log(det(Y)) + p;
% 
% obj(X1)
% dobj(W)
% dobj(inv(X))
% 
% norm(W-inv(X))
% 
% figure(1)
% plot(0:length(fvals)-1, fvals)
% hold on
% plot(0:length(dvals)-1, dvals)
% hold off
%verify_glasso
%norm(W*X - eye(p))



% p = 20;
% X = randn(25, p);
% S = cov(X);
% r = 0.5;
% maxiter = 1000*p;
% tol = 0;
% 
% 
% [X1, W, fvals, dvals] = dpglasso(S, r, maxiter, 1e-4, true);



%verify_glasso
%dvals(length(dvals)) - fvals(length(fvals))


% figure(1)
% plot(0:length(fvals)-1, fvals)
% hold on
% plot(0:length(dvals)-1, dvals)
% hold off
% 
% fvals(length(fvals))
% dvals(length(dvals))
% 
% abs(dvals(length(dvals)) - fvals(length(fvals)))



%
% Verify the lasso algorithm
%

% p = 400;
% A = randn(p);
% A = A'*A + eye(p);
% b = randn(p,1);
% x0 = randn(p,1);
% r = 0.1;
% maxiter = 1000;
% tol = 1e-8;
% 
% verify_lasso
% 
% [x1, fvals] = lasso(A, b, r, x0, maxiter, tol, true);
% 
% norm(x1 - x)
% plot(fvals)
