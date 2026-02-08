function cls = CollapseBinaryVectorToClass(target)
% target is 2 x N one-hot. Returns 1 x N class labels 0/1.
    [~, idx] = max(target, [], 1);
    cls = idx - 1;
end
