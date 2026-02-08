clear; clc; close all;
rng(0);

% Reward matrix R (from slide): -1 invalid, 0 valid, 100 to goal
R = [ -1 -1 -1 -1  0 -1;
      -1 -1 -1  0 -1 100;
      -1 -1 -1  0 -1 -1;
      -1  0  0 -1  0 -1;
       0 -1 -1  0 -1 100;
      -1  0 -1 -1  0 100 ];

nStates = size(R,1);
Q = zeros(nStates);        % Step 1: initialize Q with zeros
gamma = 0.8;               % discount factor (matches slide example)
nEpisodes = 5000;          % increase if needed (e.g., 1e4)

goal = 5;                  % goal state is "5" in the slide
startState = 2;            % start is "2" in the slide

% -------- TRAINING (Q-learning) --------
for ep = 1:nEpisodes
    s = randi([0, nStates-1]);     % random initial state (0..5)

    while s ~= goal
        validA = find(R(s+1,:) >= 0) - 1;      % actions are next-states
        a = validA(randi(numel(validA)));      % random possible action

        s_next = a;
        validNext = find(R(s_next+1,:) >= 0) - 1;

        Q(s+1, a+1) = R(s+1, a+1) + gamma * max(Q(s_next+1, validNext+1));
        s = s_next;
    end
end

disp("Learned Q matrix:");
disp(Q);

% -------- GREEDY POLICY (best path) --------
path = startState;
s = startState;

maxSteps = 20; % safety to prevent infinite loops
for k = 1:maxSteps
    if s == goal, break; end

    validA = find(R(s+1,:) >= 0) - 1;
    qvals = Q(s+1, validA+1);

    % choose best action; break ties randomly
    best = validA(qvals == max(qvals));
    a = best(randi(numel(best)));

    s = a;
    path(end+1) = s; %#ok<SAGROW>
end

disp("Greedy path from start to goal:");
disp(path);
