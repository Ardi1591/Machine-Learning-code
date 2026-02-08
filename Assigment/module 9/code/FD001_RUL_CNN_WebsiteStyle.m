clear; clc; close all;
rng(0);

%% ---------------- USER SETTINGS ----------------
trainFile = "train_SG001.txt";
testFile  = "test_SG001.txt";
rulFile   = "RUL_SG001.txt";

capRUL        = 125;    % SG001 often uses 125 
maxEpochs     = 150;
miniBatchSize = 64;

% Windowing 
winLength = 30;
winStep   = 1;

makeMonotonePred = true;   % makes predicted curve more realistic
smoothK          = 9;      % moving mean smoothing length

%% ---------------- LOAD DATA ----------------
rawTrain = localLoadDataOES(trainFile);
rawTest  = localLoadDataOES(testFile, rulFile);

disp(head(rawTrain.X{1},8));
disp(rawTrain.Y{1}(1:8));

%% ---------------- VISUALIZE (SG001 channels) ----------------
figure;
allVars = rawTrain.X{1}.Properties.VariableNames;
predictorVars = setdiff(allVars, ["id","timeStamp"]);

% plot a few channels (you can change these indices)
pick = [1 2 3 10 50 100 200 500];
pick = pick(pick <= numel(predictorVars));
varsToPlot = predictorVars(pick);

Tplot = rawTrain.X{1}(:, ['timeStamp', varsToPlot]);
stackedplot(Tplot, 'XVariable','timeStamp');
title("Example predictors (Train unit #1)");

%% ---------------- FEATURE FILTER (remove near-constant channels) ----------------
% Compute std over all training samples
Xcat = [];
for i = 1:height(rawTrain)
    Xi = rawTrain.X{i}{:, 3:end};  % channels only
    Xcat = [Xcat; Xi]; %#ok<AGROW>
end
s = std(Xcat, 0, 1);
keep = s > 1e-8;

% final channel list
chanNames = rawTrain.X{1}.Properties.VariableNames(3:end);
chanKeep  = chanNames(keep);

%% ---------------- BUILD numeric cells (T x F) and clip RUL ----------------
XTrainCell = cell(height(rawTrain),1);
YTrainCell = cell(height(rawTrain),1);
for i = 1:height(rawTrain)
    XTrainCell{i} = rawTrain.X{i}{:, chanKeep};
    YTrainCell{i} = min(rawTrain.Y{i}, capRUL);
end

XTestCell = cell(height(rawTest),1);
YTestCell = cell(height(rawTest),1);
for i = 1:height(rawTest)
    XTestCell{i} = rawTest.X{i}{:, chanKeep};
    YTestCell{i} = min(rawTest.Y{i}, capRUL);
end

%% ---------------- NORMALIZE (z-score) ----------------
Xall = vertcat(XTrainCell{:});
mu = mean(Xall, 1);
sg = std(Xall, 0, 1);
sg(sg==0) = 1;

for i = 1:numel(XTrainCell)
    XTrainCell{i} = (XTrainCell{i} - mu) ./ sg;
end
for i = 1:numel(XTestCell)
    XTestCell{i} = (XTestCell{i} - mu) ./ sg;
end

%% ---------------- TRAIN/VAL split by UNIT (avoid leakage) ----------------
nUnits = numel(XTrainCell);
idx = randperm(nUnits);
valUnits = max(1, round(0.15*nUnits));

idxVal = idx(1:valUnits);
idxTr  = idx(valUnits+1:end);

XTrUnits = XTrainCell(idxTr);
YTrUnits = YTrainCell(idxTr);

XValUnits = XTrainCell(idxVal);
YValUnits = YTrainCell(idxVal);

%% ---------------- WINDOWING (increase samples) ----------------
[XTr, YTr]   = makeWindows(XTrUnits,  YTrUnits,  winLength, winStep);
[XVal, YVal] = makeWindows(XValUnits, YValUnits, winLength, winStep);

% Convert to F x T (CTB format expects C=features, T=time)
for i = 1:numel(XTr),  XTr{i}  = XTr{i}';  YTr{i}  = YTr{i}';  end
for i = 1:numel(XVal), XVal{i} = XVal{i}'; YVal{i} = YVal{i}'; end

numFeatures = size(XTr{1},1);

%% ---------------- NETWORK (causal + dilations helps) ----------------
layers = [
    sequenceInputLayer(numFeatures)

    convolution1dLayer(3, 32, Padding="causal", DilationFactor=1)
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(3, 64, Padding="causal", DilationFactor=2)
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(3, 128, Padding="causal", DilationFactor=4)
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(3, 256, Padding="causal", DilationFactor=8)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(1)
];

options = trainingOptions("adam", ...
    InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=25, ...
    LearnRateDropFactor=0.5, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    GradientThreshold=1, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Metrics="rmse", ...
    ValidationData={XVal,YVal}, ...
    ValidationFrequency=50, ...
    InputDataFormats="CTB", ...
    Verbose=0);

net = trainnet(XTr, YTr, layers, "mse", options);

%% ---------------- TEST: windowed prediction + stitching ----------------
predictions = table("Size",[numel(XTestCell) 3], ...
    "VariableTypes", ["cell","cell","double"], ...
    "VariableNames", ["Y","YPred","RMSE"]);

for i = 1:numel(XTestCell)
    X = XTestCell{i};    % T x F
    Ytrue = YTestCell{i};    % T x 1
    T = size(X,1);

    [yPred] = stitchPredict(net, X, winLength, winStep);

    % clamp + smooth + monotone
    yPred = max(0, min(capRUL, yPred));
    yPred = movmean(yPred, smoothK);

    if makeMonotonePred
        yPred = enforceNonIncreasing(yPred);
    end

    predictions.Y{i}     = Ytrue(:)';      % 1 x T
    predictions.YPred{i} = yPred(:)';  % 1 x T
    predictions.RMSE(i)  = sqrt(mean((Ytrue(:)' - yPred(:)').^2));
end

%% ---------------- PLOTS (website-style) ----------------
figure;
histogram(predictions.RMSE, 10);
title("RMSE (Mean: " + round(mean(predictions.RMSE),2) + ", Std: " + round(std(predictions.RMSE),2) + ")");
xlabel("RMSE"); ylabel("Frequency"); grid on;

figure;
localLambdaPlot(predictions, "random");

%% ---------------- Helpers ----------------
function [Xw, Yw] = makeWindows(Xcell, Ycell, winLength, winStep)
Xw = {};
Yw = {};
for i = 1:numel(Xcell)
    X = Xcell{i};  % T x F
    Y = Ycell{i};  % T x 1
    T = size(X,1);

    % normal windows
    for t = 1:winStep:(T - winLength + 1)
        Xw{end+1,1} = X(t:t+winLength-1, :); %#ok<AGROW>
        Yw{end+1,1} = Y(t:t+winLength-1);    %#ok<AGROW>
    end

    % extra tail windows (repeat last part to emphasize low RUL)
    tailStart = max(1, T - 5*winLength);
    for t = tailStart:winStep:(T - winLength + 1)
        Xw{end+1,1} = X(t:t+winLength-1, :); %#ok<AGROW>
        Yw{end+1,1} = Y(t:t+winLength-1);    %#ok<AGROW>
    end
end
end

function yPred = stitchPredict(net, X, winLength, winStep)
% X: T x F
T = size(X,1);
predSum = zeros(1,T);
predCnt = zeros(1,T);

for t = 1:winStep:(T - winLength + 1)
    segX = X(t:t+winLength-1, :)';   % F x winLength
    segY = minibatchpredict(net, segX, MiniBatchSize=1, InputDataFormats="CTB"); % 1 x winLength

    predSum(t:t+winLength-1) = predSum(t:t+winLength-1) + segY;
    predCnt(t:t+winLength-1) = predCnt(t:t+winLength-1) + 1;
end

yPred = predSum ./ max(predCnt,1);
end

function y = enforceNonIncreasing(y)
y = y(:)';
for k = 2:numel(y)
    if y(k) > y(k-1)
        y(k) = y(k-1);
    end
end
end


