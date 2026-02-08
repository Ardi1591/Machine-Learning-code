clear; clc; close all;
rng(0);

% ---------------- USER SETTINGS ----------------
trainFile = "train_SG001.txt";
testFile  = "test_SG001.txt";
rulFile   = "RUL_SG001.txt";

capRUL        = 125;     % like your project (MathWorks example uses 150)
maxEpochs     = 40;
miniBatchSize = 16;

% ---------------- LOAD DATA ----------------
rawTrain = localLoadDataOES(trainFile);
rawTest  = localLoadDataOES(testFile, rulFile);

% ---------------- REMOVE LOW-VARIABILITY FEATURES ----------------
% MathWorks uses prognosability(...,"timeStamp"). If you don't have it, fallback to variance filter.
if exist("prognosability","file") == 2
    prog = prognosability(rawTrain.X, "timeStamp"); % Predictive Maintenance Toolbox
    idxToRemove = prog.Variables == 0 | isnan(prog.Variables);
    featToRetain = prog.Properties.VariableNames(~idxToRemove);
else
    % Fallback: keep channels with non-trivial std over all training samples
    allX = [];
    for i = 1:height(rawTrain)
        Xi = rawTrain.X{i}{:,3:end}; % channels only
        allX = [allX; Xi]; %#ok<AGROW>
    end
    s = std(allX, 0, 1);
    featToRetain = "Channel_" + string(find(s > 1e-10));
end

% Convert each X{i} from table -> numeric matrix using retained features
for i = 1:height(rawTrain)
    rawTrain.X{i} = rawTrain.X{i}{:, featToRetain};
end
for i = 1:height(rawTest)
    rawTest.X{i}  = rawTest.X{i}{:, featToRetain};
end

% ---------------- NORMALIZE TRAINING PREDICTORS (z-score) ----------------
[~, Xmu, Xsigma] = zscore(vertcat(rawTrain.X{:}));
Xsigma(Xsigma==0) = 1; % safety

preTrain = table();
preTrain.X = cell(height(rawTrain),1);
preTrain.Y = cell(height(rawTrain),1);

for i = 1:height(rawTrain)
    preTrain.X{i} = (rawTrain.X{i} - Xmu) ./ Xsigma;
    preTrain.Y{i} = min(rawTrain.Y{i}, capRUL);
end

% Apply SAME normalization + clipping to test
for i = 1:height(rawTest)
    rawTest.X{i} = (rawTest.X{i} - Xmu) ./ Xsigma;
    rawTest.Y{i} = min(rawTest.Y{i}, capRUL);
end

% ---------------- PREPARE DATA FOR PADDING (features x time) ----------------
sequenceLengths = zeros(height(preTrain),1);
for i = 1:height(preTrain)
    preTrain.X{i} = preTrain.X{i}';   % features x time
    preTrain.Y{i} = preTrain.Y{i}';   % 1 x time
    sequenceLengths(i) = size(preTrain.X{i}, 2);
end

% Sort by length (reduces padding)
[~, idx] = sort(sequenceLengths, "descend");
XTrain = preTrain.X(idx);
YTrain = preTrain.Y(idx);

% ---------------- NETWORK ARCHITECTURE (MathWorks-style causal 1D CNN) ----------------
numFeatures = size(XTrain{1},1);
numHiddenUnits = 100;
numResponses = 1;

layers = [
    sequenceInputLayer(numFeatures)

    convolution1dLayer(5,  32, Padding="causal")
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(7,  64, Padding="causal")
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(11, 128, Padding="causal")
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(13, 256, Padding="causal")
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(15, 512, Padding="causal")
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numHiddenUnits)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(numResponses)
];

% ---------------- TRAIN ----------------
options = trainingOptions("adam", ...
    LearnRateSchedule="piecewise", ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=0.01, ...
    GradientThreshold=1, ...
    Shuffle="never", ...                 % keep sorted sequences
    Plots="training-progress", ...
    Metrics="rmse", ...
    InputDataFormats="CTB", ...          % C=features, T=time, B=batch
    Verbose=0);

net = trainnet(XTrain, YTrain, layers, "mse", options);

% ---------------- TEST ----------------
predictions = table("Size",[height(rawTest) 3], ...
    "VariableTypes", ["cell","cell","double"], ...
    "VariableNames", ["Y","YPred","RMSE"]);

for i = 1:height(rawTest)
    unitX = rawTest.X{i}';       % features x time
    unitY = rawTest.Y{i}';       % 1 x time

    yPred = minibatchpredict(net, unitX, MiniBatchSize=1, InputDataFormats="CTB");

    predictions.Y{i}     = unitY;
    predictions.YPred{i} = yPred;
    predictions.RMSE(i)  = sqrt(mean((unitY - yPred).^2));
end

figure;
histogram(predictions.RMSE, NumBins=10);
title("RMSE (Mean: " + round(mean(predictions.RMSE),2) + ", Std: " + round(std(predictions.RMSE),2) + ")");
xlabel("RMSE"); ylabel("Frequency");

% Plot one unit (random)
figure;
localLambdaPlot(predictions, "random");

% ---------------- Helper ----------------
function localLambdaPlot(predictions, lambdaCase)
    if isnumeric(lambdaCase)
        idx = lambdaCase;
    else
        switch lower(string(lambdaCase))
            case {"random","r"}
                idx = randi(height(predictions));
            case {"best","b"}
                [~,idx] = min(predictions.RMSE);
            case {"worst","w"}
                [~,idx] = max(predictions.RMSE);
            otherwise
                idx = 1;
        end
    end

    y     = predictions.Y{idx};
    yPred = predictions.YPred{idx};

    x = 0:numel(y)-1;
    plot(x, y, x, yPred);
    legend("True RUL","Predicted RUL");
    xlabel("Time stamp (sequence index)");
    ylabel("RUL (cycles)");
    title("RUL for Test unit #" + idx + " (" + lambdaCase + ")");
end
