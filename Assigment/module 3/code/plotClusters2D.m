function plotClusters2D(X, idx, C)
    K = size(C,1);
    hold on;

    % Plot points by cluster (markers similar to slide style)
    markers = {'o','*','x','+','s','d'};
    for k = 1:K
        pts = X(idx==k,:);
        scatter(pts(:,1), pts(:,2), 10, markers{min(k,numel(markers))});
    end

    % Plot centroids
    for k = 1:K
        plot(C(k,1), C(k,2), 'kx', 'MarkerSize', 14, 'LineWidth', 3);
    end

    grid on; xlabel('x-value'); ylabel('y-value');

    % Legend
    leg = strings(1, K+1);
    for k = 1:K
        leg(k) = "Cluster " + k;
    end
    leg(K+1) = "Centroids";
    legend(leg, 'Location','best');
end