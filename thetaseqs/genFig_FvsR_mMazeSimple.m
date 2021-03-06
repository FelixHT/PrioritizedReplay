%% STATE-SPACE PARAMETERS
addpath('../../../');
clear;
setParams;

% double size end arm maze
params.maze             = zeros(6,11); % zeros correspond to 'visitable' states
params.maze(2:6,3:9)      = 1; % wall
params.maze(2:4,6)      = 0; % path in the center
params.s_start          = [4,6]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal
params.s_end            = [6,1]; % goal state (in matrix notation)
params.s_choice = [1,6];
params.rewMag           = 1; % reward magnitude (rows: locations; columns: values)

dists_choice = [5 6 7 8 9 10, 4 5 6 7 8 9, 3 nan nan nan nan nan 2 nan nan nan nan nan 1 nan nan nan nan nan 0 -1 -2 -3 nan nan 1 nan nan nan nan nan 2 nan nan nan nan nan 3 nan nan nan nan nan 4 5 6 7 8 9, 5 6 7 8 9 10];
dists_start = dists_choice + 3;

%% SIMPLE MAZE
% params.maze             = zeros(6,7); % zeros correspond to 'visitable' states
% params.maze(2:6,2:3)      = 1; % wall
% params.maze(2:6,5:6)      = 1; % wall
% params.s_start          = [6,4]; % beginning state (in matrix notation)
start_position = sub2ind(size(params.maze), params.s_start(1),params.s_start(2));
% params.s_start_rand     = false; % Start at random locations after reaching goal
% params.s_end            = [6,1]; % goal state at the left
% params.rewMag           = [1]; % reward magnitude (rows: locations; columns: values)
above_end_position = sub2ind(size(params.maze), params.s_end(1)-1,params.s_end(2));  % one position above the goal
%% params.s_end            = [6,1;6,7]; % goal state at left and right (in matrix notation)
%% above_end_1_position = sub2ind(size(params.maze), params.s_end(1,1)-1,params.s_end(1,2))
%% above_end_2_position = sub2ind(size(params.maze), params.s_end(2,1)-1,params.s_end(2,2))
% params.s_choice = [2,4];
choice_position = sub2ind(size(params.maze), params.s_choice(1),params.s_choice(2));


% params.rewMag           = [1; 1]; % reward magnitude (rows: locations; columns: values)
params.rewSTD           = 0.1; % reward Gaussian noise (rows: locations; columns: values)
params.rewProb          = 1; % probability of receiving each reward (columns: values)
params.planAtChoicePoint = true;



params.planOnlyAtGorS   = true;


%% OVERWRITE PARAMETERS
params.N_SIMULATIONS    = 50; % number of times to run the simulation
params.MAX_N_STEPS      = 1e5; % maximum number of steps to simulate
params.MAX_N_EPISODES   = 50; % maximum number of episodes to simulate (use Inf if no max)
params.nPlan            = 20; % number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as it is worth it)
params.onVSoffPolicy    = 'on-policy'; % Choose 'off-policy' (default, learns Q*) or 'on-policy' (learns Qpi) learning for updating Q-values and computing gain

params.alpha            = 1.0; % learning rate
params.gamma            = 0.9; % discount factor
params.softmaxInvT      = 5; % soft-max inverse temperature temperature
params.tieBreak         = 'min'; % How to break ties on EVM (choose which sequence length is prioritized: 'min', 'max', or 'rand')
params.setAllGainToOne  = false; % Set the gain term of all items to one (for debugging purposes)
params.setAllNeedToOne  = false; % Set the need term of all items to one (for debugging purposes)
params.setAllNeedToZero = false; % Set the need term of all items to zero, except for the current state (for debugging purposes)

enablePlotting=false;
params.PLOT_STEPS       = enablePlotting; % Plot each step of real experience
params.PLOT_Qvals       = enablePlotting; % Plot Q-values
params.PLOT_PLANS       = enablePlotting; % Plot each planning step
params.PLOT_EVM         = false; % Plot need and gain
params.PLOT_wait        = 50 ; % Number of full episodes completed before plotting

% saveStr = input('Do you want to produce figures (y/n)? ','s');
% if strcmp(saveStr,'y')
%     saveBool = true;
% else
%     saveBool = false;
% end


%% RUN SIMULATION
rng(mean('replay'));
for k=1:params.N_SIMULATIONS
    simData(k) = replaySim(params);
end


%% ANALYSIS PARAMETERS
minNumCells = 5;
minFracCells = 0;
runPermAnalysis = true; % Run permutation analysis (true or false)
nPerm = 500; % Number of permutations for assessing significance of an event


%% INITIALIZE VARIABLES
forwardCount = zeros(length(simData),numel(params.maze));
reverseCount = zeros(length(simData),numel(params.maze));
nextState = nan(numel(params.maze),4);


%% RUN ANALYSIS

% Get action consequences from stNac2stp1Nr()
for s=1:numel(params.maze)
    [I,J] = ind2sub(size(params.maze),s);
    st=nan(1,2);
    st(1)=I; st(2) = J;
    for a=1:4
        [~,~,stp1i] = stNac2stp1Nr(st,a,params);
        nextState(s,a) = stp1i;
    end
end
%%

significantReplays = cell(params.N_SIMULATIONS, 0);  % cell used to store significant replay events
replayDirections = cell(params.N_SIMULATIONS, 0);    % cell used to store direction of significant replay events (forward vs reverse)
agentPosInSignSeq = cell(params.N_SIMULATIONS, 0);    % cell used to store the agent's position at each significant replay
agentPosBefSignSeq = cell(params.N_SIMULATIONS, 0);    % cell used to store the agent's position before each significant replay
allPlanningSteps = cell(params.N_SIMULATIONS, 0);     % cell to store all planning steps
agentPosPlanning = cell(params.N_SIMULATIONS, 0);     % cell used to store the agent's position at each planning
agentPosBefPlanning = cell(params.N_SIMULATIONS, 0);     % cell used to store the agent's position before each planning

for k=1:length(simData)  % for each simulation
% for k=1:params.N_SIMULATIONS
    fprintf('Simulation #%d\n',k);
    % Identify candidate replay events: timepoints in which the number of replayed states is greater than max(valid positions *
    % minFracCells,minNumCells), but here minFracCells is set to zero, so it just needs to be bigger than minNumCells. This will 
    % happen at every planning position, because params.nPlan is bigger than minNumCells
    candidateEvents = find(cellfun('length',simData(k).replay.state)>=max(sum(params.maze(:)==0)*minFracCells,minNumCells));  % indexes of ts where there was a replay
    lapNum = [0;simData(k).numEpisodes(1:end-1)] + 1; % episode number for each time point
    lapNum_events = lapNum(candidateEvents); % episode number for each candidate event
    agentPos = simData(k).expList(candidateEvents,1); % agent position during each candidate event
    agentPosBef = simData(k).expList(candidateEvents-1,1); % agent position immediately before each candidate event
    countSignReplays = 0;  % number of significant replays
    
    for e=1:length(candidateEvents)  % for each candidate event
        eventState = simData(k).replay.state{candidateEvents(e)}; % In a multi-step sequence, simData.replay.state has 1->2 in one row, 2->3 in another row, etc
        eventAction = simData(k).replay.action{candidateEvents(e)}; % In a multi-step sequence, simData.replay.action has the action taken at each step of the trajectory
        
        allPlanningSteps{k, e}=eventState;
        agentPosPlanning{k, e}= agentPos(e);
        agentPosBefPlanning{k, e}=agentPosBef(e);
        
        % Identify break points in this event, separating event into sequences
        eventDir = cell(1,length(eventState)-1);
        breakPts = 0; % Save breakpoints that divide contiguous replay events
        for i=1:(length(eventState)-1)
            % If state(i) and action(i) leads to state(i+1): FORWARD
            if nextState(eventState(i),eventAction(i)) == eventState(i+1)
                eventDir{i} = 'F';
            end
            % If state(i+1) and action(i+1) leads to state(i): REVERSE
            if nextState(eventState(i+1),eventAction(i+1)) == eventState(i)
                eventDir{i} = 'R';
            end
            
            % Find if this is a break point
            if isempty(eventDir{i}) % If this transition was neither forward nor backward
                breakPts = [breakPts (i-1)]; % Then, call this a breakpoint
            elseif i>1
                if ~strcmp(eventDir{i},eventDir{i-1}) % If this transition was forward and the previous was backwards (or vice-versa)
                    breakPts = [breakPts (i-1)]; % Then, call this a breakpoint
                end
            end
            if i==(length(eventState)-1)
                breakPts = [breakPts i]; % Add a breakpoint after the last transition
            end
        end
        
        % Break this event into segments of sequential activity and check
        % their statistical significance
        for j=1:(numel(breakPts)-1)
            thisChunk = (breakPts(j)+1):(breakPts(j+1));
            if (length(thisChunk)+1) >= minNumCells
                % Extract information from this sequential event
                replayDir = eventDir(thisChunk); % Direction of transition
                %disp(replayDir)
                %disp(length(replayDir))
                replayState = eventState([thisChunk (thisChunk(end)+1)]); % replayState is the set of states of a sequence from Start state to end
                replayAction = eventAction([thisChunk (thisChunk(end)+1)]); % Action
                
                % Assess the significance of this event
                %allPerms = cell2mat(arrayfun(@(x)randperm(length(replayState)),(1:nPerm)','UniformOutput',0));
                sigBool = true; %#ok<NASGU>
                if runPermAnalysis
                    fracFor = nanmean(strcmp(replayDir,'F')); % Fraction of transitions in this chunk whose direction was forward
                    fracRev = nanmean(strcmp(replayDir,'R')); % Fraction of transitions in this chunk whose direction was reverse
                    disScore = fracFor-fracRev;
                    dirScore_perm = nan(1,nPerm);
                    for p=1:nPerm
                        thisPerm = randperm(length(replayState));
                        replayState_perm = replayState(thisPerm);
                        replayAction_perm = replayAction(thisPerm);
                        replayDir_perm = cell(1,length(replayState_perm)-1);
                        for i=1:(length(replayState_perm)-1)
                            if nextState(replayState_perm(i),replayAction_perm(i)) == replayState_perm(i+1)
                                replayDir_perm{i} = 'F';
                            end
                            if nextState(replayState_perm(i+1),replayAction_perm(i+1)) == replayState_perm(i)
                                replayDir_perm{i} = 'R';
                            end
                        end
                        fracFor = nanmean(strcmp(replayDir_perm,'F'));
                        fracRev = nanmean(strcmp(replayDir_perm,'R'));
                        dirScore_perm(p) = fracFor-fracRev;
                    end
                    dirScore_perm = sort(dirScore_perm);
                    lThresh_score = dirScore_perm(floor(nPerm*0.025));  % p-value of 0.05 double-sided
                    hThresh_score = dirScore_perm(ceil(nPerm*0.975));
                    if (disScore<lThresh_score) || (disScore>hThresh_score)
                        sigBool = true;
                    else
                        sigBool = false;
                    end
                end
                
                % Add significant events to 'bucket'
                if sigBool
                    countSignReplays  = countSignReplays +1;
%                     disp(replayState);
%                     disp(agentPos(e));
%                     disp(replayDir{1});
                    significantReplays{k, countSignReplays}=replayState;
                    agentPosInSignSeq{k, countSignReplays}= agentPos(e);
                    agentPosBefSignSeq{k, countSignReplays}= agentPosBef(e);
                    if replayDir{1}=='F'
%                         replayDirections{k, countSignReplays}='F';
                        forwardCount(k,agentPos(e)) = forwardCount(k,agentPos(e)) + 1;
                    elseif replayDir{1}=='R'
%                         replayDirections{k, countSignReplays}='R';
                        reverseCount(k,agentPos(e)) = reverseCount(k,agentPos(e)) + 1;
                    end
                end
            end
        end
    end
end

all_choice_lens = [];
all_choice_lens_fwd = [];
all_start_lens = [];

all_choice_dists = cell(0);
all_start_dists = cell(0);

for run=1:params.N_SIMULATIONS
    % in seqs
    idxs_choice_signSeq = find(cellfun(@(x)isequal(x,choice_position), agentPosInSignSeq(run,:)));  % find sequences when the agent is at the choice position
    idxs_choice_signSeq_fwd = find(cellfun(@(x,y)and(isequal(x,choice_position),isequal(y,choice_position+1)), agentPosInSignSeq(run,:), agentPosBefSignSeq(run,:)));  % find sequences when the agent is at the choice position and is going forward (came from the position below)
    idxs_start_signSeq = find(cellfun(@(x)isequal(x,start_position), agentPosInSignSeq(run,:)));  % find sequences when the agent is at the start position

    choice_replay_lengths = cellfun(@(x) size(x,2), significantReplays(run,idxs_choice_signSeq));
    choice_replay_lengths_fwd = cellfun(@(x) size(x,2), significantReplays(run,idxs_choice_signSeq_fwd));
    start_replay_lengths = cellfun(@(x) size(x,2), significantReplays(run,idxs_start_signSeq));
    
%     mean(choice_replay_lengths);
%     mean(start_replay_lengths);
    
    all_choice_lens = [all_choice_lens choice_replay_lengths];
    all_choice_lens_fwd = [all_choice_lens_fwd choice_replay_lengths_fwd];
    all_start_lens = [all_start_lens start_replay_lengths];
    
    % in planning
    idxs_choice_planning = find(cellfun(@(x,y)and(isequal(x,choice_position),isequal(y,choice_position+1)), agentPosPlanning(run,:), agentPosBefPlanning(run,:)));
%     idxs_choice_planning = find(cellfun(@(x)isequal(x,choice_position), agentPosPlanning(run,:)));
    idxs_start_planning = find(cellfun(@(x)isequal(x,start_position), agentPosPlanning(run,:)));
    
    choice_planning_dists = cellfun(@(x) dists_choice(x), allPlanningSteps(run,idxs_choice_planning), 'UniformOutput', false);
    start_planning_dists = cellfun(@(x) dists_start(x), allPlanningSteps(run,idxs_start_planning), 'UniformOutput', false);
    
    concat_choice_dists=[];
    for i=1:size(choice_planning_dists,2)
        concat_choice_dists = [concat_choice_dists choice_planning_dists{i}];
    end
    
    concat_start_dists=[];
    for i=1:size(start_planning_dists,2)
        concat_start_dists = [concat_start_dists start_planning_dists{i}];
    end
    
    all_choice_dists{run} = concat_choice_dists;
    all_start_dists{run} = concat_start_dists;
end


% figure(10); clf;
% hold on
% data = [mean(all_choice_lens), mean(all_start_lens)];
% err = [std(all_choice_lens)/sqrt(params.N_SIMULATIONS), std(all_start_lens)/sqrt(params.N_SIMULATIONS)];
% f1 = bar(data);
% errorbar(1:2,data,err,err);
% title('Length of sequences');
% xticks([1,2]);
% set(f1(1).Parent,'XTickLabel',{'Choice point','Start point'});


figure(1); clf;
hold on
data = [mean(all_choice_lens_fwd),mean(all_start_lens)];
err = [std(all_choice_lens_fwd)/sqrt(params.N_SIMULATIONS), std(all_start_lens)/sqrt(params.N_SIMULATIONS)];
f10 = bar(data);
errorbar(1:2,data,err,err);
title('Length of sequences');
xticks([1,2]);
set(f10(1).Parent,'XTickLabel',{'Choice point','Start point'});

figure(2); clf;
% run = 8
across_runs_choice_dists = cat(2, all_choice_dists{:});
choice_hist = histogram(across_runs_choice_dists, "Normalization", "probability");
title('Choice point');
figure(3); clf;
across_runs_start_dists = cat(2, all_start_dists{:});
% start_hist = histogram(all_start_dists{run}, "Normalization", "probability");
start_hist = histogram(across_runs_start_dists, "Normalization", "probability");
title('Start point');

figure(4); clf;
diff = [choice_hist.Values 0 0]-[0 0 0 start_hist.Values];
% diff = [choice_hist.Values 0 0 0]-[0 0 0 start_hist.Values];
f4 = bar(diff);
xticks(1:16);
set(f4(1).Parent,'XTickLabel',-3:12);
title('Choice - Start');


% % TO VISUALIZE SIGNIFICANT SEQUENCES FROM A SPECIFIC RUN STARTING AT THE
% % CHOICE POINT COMING FROM THE CENTRAL ARM  - BEGIN
% run = 1
% idxs_choice_signSeq_fwd = find(cellfun(@(x,y)and(isequal(x,choice_position),isequal(y,choice_position+1)), agentPosInSignSeq(run,:), agentPosBefSignSeq(run,:)));  % find sequences when the agent is at the choice position and is going forward (came from the position below)
% myreplays = significantReplays(run,idxs_choice_signSeq_fwd)
% for i=1:numel(myreplays)
%     disp(myreplays{i})
% end
% % TO VISUALIZE SIGNIFICANT SEQUENCES FROM A SPECIFIC RUN STARTING AT THE
% % CHOICE POINT COMING FROM THE CENTRAL ARM  - END



% % Compute the number of significant events BEFORE (preplay) and AFTER (replay) an event (which could be larger than 1)
% % PS: Notice that this is not a measure of the percent of episodes with a significant event (which would produce a smaller numbers)
% preplayF = nansum([forwardCount(:,1),forwardCount(:,30)],2)./params.MAX_N_EPISODES;
% replayF = nansum([forwardCount(:,6),forwardCount(:,25)],2)./params.MAX_N_EPISODES;
% preplayR = nansum([reverseCount(:,1),reverseCount(:,30)],2)./params.MAX_N_EPISODES;
% replayR = nansum([reverseCount(:,6),reverseCount(:,25)],2)./params.MAX_N_EPISODES;
% 
% % count number of forward and reverse replays at the choice point
% choiceF = nansum([forwardCount(:,choice_position),forwardCount(:,choice_position)],2)./params.MAX_N_EPISODES;
% choiceR = nansum([reverseCount(:,choice_position),reverseCount(:,choice_position)],2)./params.MAX_N_EPISODES;
%% PLOT

% % Forward-vs-Reverse
% figure(1); clf;
% f1 = bar([nanmean(preplayF) nanmean(replayF) ; nanmean(preplayR) nanmean(replayR)]);
% legend({'Preplay','Replay'},'Location','NortheastOutside');
% f1(1).FaceColor=[1 1 1]; % Replay bar color
% f1(1).LineWidth=1;
% f1(2).FaceColor=[0 0 0]; % Replay bar color
% f1(2).LineWidth=1;
% set(f1(1).Parent,'XTickLabel',{'Forward correlated','Reverse correlated'});
% ymax=ceil(max(reshape([nanmean(preplayF) nanmean(replayF) ; nanmean(preplayR) nanmean(replayR)],[],1)));
% ylim([0 ymax]);
% ylabel('Events/Lap');
% grid on
