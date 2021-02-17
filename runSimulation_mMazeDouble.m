%% STATE-SPACE PARAMETERS
setParams;

% two m mazes with 2 starting points and 2 goals
params.maze             = zeros(6,15); % zeros correspond to 'visitable' states
params.maze(2:6,2:3)      = 1; % wall
params.maze(2:6,5:6)      = 1; % wall
params.maze(1:6,8)          = 1; % wall
params.maze(2:6,10:11)      = 1; % wall
params.maze(2:6,13:14)      = 1; % wall
params.s_start          = [6,4; 6,12]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal
params.s_end            = [6,1;6,15]; % [6,1;6,7;6,9;6,15]; goal state (in matrix notation)
params.rewMag           = [1;1]; % reward magnitude (rows: locations; columns: values)

params.s_choice = [2,4];

% linear track 2 starting locations, 2 goals
% params.maze             = zeros(3,10); % zeros correspond to 'visitable' states
% params.maze(2,:)        = 1; % wall
% params.s_start          = [1,1;3,size(params.maze,2)]; % beginning state (in matrix notation)
% params.s_start_rand     = false; % Start at random locations after reaching goal
% 
% params.s_end            = [1,size(params.maze,2);3,1]; % goal state (in matrix notation)

%params.s_end            = [1,9;6,9]; % goal state (in matrix notation)
%params.rewMag           = [1 0.5; 0.1 0.05]; % reward magnitude (rows: locations; columns: values)
%params.rewSTD           = [1 0.5; 0.1 0.05]; % reward Gaussian noise (rows: locations; columns: values)
% params.rewMag           = [1,0;0,1;0,1;1,0]; % reward magnitude (rows: locations; columns: values)
params.rewSTD           = 0.1; % reward Gaussian noise (rows: locations; columns: values)
params.rewProb          = 1; % probability of receiving each reward (columns: values)

%% PLOTTING SETTINGS
params.PLOT_STEPS       = true; % Plot each step of real experience
params.PLOT_Qvals       = true; % Plot Q-values
params.PLOT_PLANS       = true; % Plot each planning step
params.PLOT_EVM         = true; % Plot need and gain
params.PLOT_TRACE       = false; % Plot all planning traces
params.PLOT_wait        = 5 ; % Number of full episodes completed before plotting


%% RUN SIMULATION
rng(mean('replay'));  % mean('replay') is the seed for the random number generator
simData = replaySim(params);
