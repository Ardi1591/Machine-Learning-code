function data = localLoadDataOES(filenamePredictors, varargin)
% localLoadDataOES  SG001/OES loader
% Output: table with variables X and Y
%   X{k} : table for unit k (id,timeStamp,Channel_*)
%   Y{k} : column vector RUL for unit k

% Optional response file
filenameResponses = "";
if ~isempty(varargin)
    filenameResponses = string(varargin{1});
end
hasResponses = strlength(filenameResponses) > 0;

% Read predictors (tab-delimited)
M = readmatrix(filenamePredictors, "FileType","text", "Delimiter","\t");

% Remove blank rows/cols
M = M(~all(isnan(M),2), :);
M(:, all(isnan(M),1)) = [];

nCols = size(M,2);
if nCols < 3
    error("SG001 expects >=3 columns: id, timeStamp, Channel_1..");
end

nChannels = nCols - 2;

rawData = array2table(M);
rawData.Properties.VariableNames = ["id","timeStamp","Channel_" + string(1:nChannels)];

% Load end-of-sequence RUL for test if provided
if hasResponses
    RULTest = readmatrix(filenameResponses, "FileType","text");
    RULTest = RULTest(:);
end

IDs  = rawData.id;
uIDs = unique(IDs, "stable");
numObs = numel(uIDs);

X = cell(numObs,1);
Y = cell(numObs,1);

if hasResponses && numel(RULTest) < numObs
    error("RUL file has %d rows but predictors contain %d units.", numel(RULTest), numObs);
end

for k = 1:numObs
    uid = uIDs(k);
    idx = (IDs == uid);

    Xi = rawData(idx,:);
    X{k} = Xi;

    if ~hasResponses
        t = Xi.timeStamp;
        Y{k} = (max(t) - t);                 % train: last timestep -> 0
    else
        seqLen = height(Xi);
        endRUL = RULTest(k);                 % test: provided end-RUL
        Y{k} = (endRUL + (seqLen-1:-1:0))';
    end
end

data = table(X, Y, 'VariableNames', {'X','Y'});
end
