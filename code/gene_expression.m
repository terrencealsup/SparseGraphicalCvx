%X = csvread('../data/preprocessed_data.csv');

%X = X(:,1:1000);

%S = cov(X);

%r = 100;
%p = size(S, 1);
%[P, W] = dpglasso(S, r, 10*p, 1e-4);