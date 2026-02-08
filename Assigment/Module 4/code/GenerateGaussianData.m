function [data, target] = GenerateGaussianData(N, Mean1, Sigma1, Mean2, Sigma2)
% GenerateGaussianData  Create 2-class 2D Gaussian dataset
%   data   : 2 x N  (columns are samples)
%   target : 2 x N  (one-hot: row1=class0, row2=class1)

    if mod(N,2) ~= 0
        nA = floor(N/2);
        nB = N - nA;
    else
        nA = N/2;
        nB = N/2;
    end

    Mean1 = Mean1(:);  Mean2 = Mean2(:);

    % Sample using Cholesky (no toolbox needed)
    L1 = chol(Sigma1, "lower");
    L2 = chol(Sigma2, "lower");

    A = L1 * randn(2, nA) + Mean1;   % class 0
    B = L2 * randn(2, nB) + Mean2;   % class 1

    data = [A B];

    target = [ones(1,nA) zeros(1,nB);
              zeros(1,nA) ones(1,nB)];

    % Shuffle samples (optional but nice)
    p = randperm(N);
    data = data(:,p);
    target = target(:,p);
end
