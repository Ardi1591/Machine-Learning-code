function localLambdaPlot(predictions, whichCase)

n = height(predictions);
if nargin < 2, whichCase = "random"; end

if isnumeric(whichCase)
    idx = whichCase;
else
    switch lower(string(whichCase))
        case "random"
            idx = randi(n);
        case "best"
            [~,idx] = min(predictions.RMSE);
        case "worst"
            [~,idx] = max(predictions.RMSE);
        otherwise
            idx = 1;
    end
end

yTrue = predictions.Y{idx}(:)';
yPred = predictions.YPred{idx}(:)';

t = 0:numel(yTrue)-1;
plot(t, yTrue, t, yPred, "LineWidth", 1.2);
legend("True RUL","Predicted RUL", "Location","northeast");
xlabel("Time stamp (sequence index)");
ylabel("RUL (cycles)");
title("RUL for Test unit #" + idx + " (" + string(whichCase) + ")");
grid on;

end
