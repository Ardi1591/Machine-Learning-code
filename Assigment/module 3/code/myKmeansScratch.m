function [idx, C, SSE_hist, idx_hist, C_hist] = myKmeansScratch(X, C0, maxIter, tol)
    % X: N x D
    % C0: K x D
    % idx: N x 1 (cluster labels)
    % SSE_hist: SSE per iteration

    C = C0;
    K = size(C,1);
    N = size(X,1);

    SSE_hist = zeros(maxIter,1);
    idx_hist = cell(maxIter,1);
    C_hist   = cell(maxIter,1);

    idx_prev = zeros(N,1);

    for it = 1:maxIter
        % ---- Assignment step ----
        % Compute squared distances to each centroid (N x K)
        D2 = zeros(N,K);
        for k = 1:K
            diff = X - C(k,:);
            D2(:,k) = sum(diff.^2, 2);
        end
        [minD2, idx] = min(D2, [], 2);

        % SSE = sum of squared distances to assigned centroid
        SSE_hist(it) = sum(minD2);

        idx_hist{it} = idx;
        C_hist{it}   = C;

        % ---- Update step ----
        C_new = C;
        for k = 1:K
            pts = X(idx==k,:);
            if ~isempty(pts)
                C_new(k,:) = mean(pts,1);
            end
            % if empty, keep old centroid
        end

        % ---- Termination checks ----
        centroidShift = max(vecnorm(C_new - C, 2, 2));  % max movement
        sameAssign = all(idx == idx_prev);

        C = C_new;
        idx_prev = idx;

        if sameAssign || centroidShift < tol
            SSE_hist = SSE_hist(1:it);
            idx_hist = idx_hist(1:it);
            C_hist   = C_hist(1:it);
            break;
        end
    end
end