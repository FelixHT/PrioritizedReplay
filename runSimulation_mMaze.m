%% STATE-SPACE PARAMETERS
setParams;
% single m maze with 2 goals
params.maze             = zeros(6,7); % zeros correspond to 'visitable' states
params.maze(2:6,2:3)      = 1; % wall
params.maze(2:6,5:6)      = 1; % wall
params.s_start          = [6,4]; % beginning state (in matrix notation)
start_position = sub2ind(size(params.maze), params.s_start(1),params.s_start(2));
params.s_start_rand     = false; % Start at random locations after reaching goal
params.s_end            = [6,1]; % goal state (in matrix notation)
above_end_position = sub2ind(size(params.maze), params.s_end(1)-1,params.s_end(2));  % one position above the goal
params.rewMag           = [1; 1]; % reward magnitude (rows: locations; columns: values)

params.s_choice = [2,4];
choice_position = sub2ind(size(params.maze), params.s_choice(1),params.s_choice(2));
params.planAtChoicePoint = true;
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
enablePlotting=false;
params.PLOT_STEPS       = enablePlotting; % Plot each step of real experience
params.PLOT_Qvals       = enablePlotting; % Plot Q-values
params.PLOT_PLANS       = enablePlotting; % Plot each planning step
params.PLOT_EVM         = enablePlotting; % Plot need and gain
params.PLOT_TRACE       = false; % Plot all planning traces
params.PLOT_wait        = 3 ; % Number of full episodes completed before plotting


%% RUN SIMULATION
rng(mean('replay'));  % mean('replay') is the seed for the random number generator
simData = replaySim(params);



%% INITIALIZE VARIABLES
forwardCount = zeros(length(simData),numel(params.maze));
reverseCount = zeros(length(simData),numel(params.maze));
nextState = nan(numel(params.maze),4);

k=1;


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

% Identify candidate replay events: timepoints (ts) in which the number of replayed states is greater than max(valid positions *
% minFracCells,minNumCells), but here minFracCells is set to zero, so it just needs to be bigger than minNumCells
candidateEvents = find(cellfun('length',simData.replay.state)>=max(sum(params.maze(:)==0)*minFracCells,minNumCells));  % indexes of ts where there was replay
lapNum = [0;simData.numEpisodes(1:end-1)] + 1; % episode number for each time point
lapNum_events = lapNum(candidateEvents); % episode number for each candidate event
agentPos = simData.expList(candidateEvents,1); % agent position during each candidate event

for e=1:length(candidateEvents)  % for each candidate event
    eventState = simData.replay.state{candidateEvents(e)}; % In a multi-step sequence, simData.replay.state has 1->2 in one row, 2->3 in another row, etc
    eventAction = simData.replay.action{candidateEvents(e)}; % In a multi-step sequence, simData.replay.action has the action taken at each step of the trajectory

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

    % Break this event into segments of sequential activity
    for j=1:(numel(breakPts)-1)  % for each chunk
        thisChunk = (breakPts(j)+1):(breakPts(j+1));  % from one break point to the next
        if (length(thisChunk)+1) >= minNumCells
            % Extract information from this sequential event
            replayDir = eventDir(thisChunk); % Direction of transition
            replayState = eventState([thisChunk (thisChunk(end)+1)]); % Start state
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
                lThresh_score = dirScore_perm(floor(nPerm*0.025));
                hThresh_score = dirScore_perm(ceil(nPerm*0.975));
                if (disScore<lThresh_score) || (disScore>hThresh_score)
                    sigBool = true;
                else
                    sigBool = false;
                end
            end

            % Add significant events to 'bucket': a matrix counting the
            % number of significant events per state
            if sigBool
                if replayDir{1}=='F'
                    forwardCount(k,agentPos(e)) = forwardCount(k,agentPos(e)) + 1;
                elseif replayDir{1}=='R'
                    reverseCount(k,agentPos(e)) = reverseCount(k,agentPos(e)) + 1;
                end
            end
        end
    end
end


% simData.replay.state{simData.numEpisodes==1}


% Compute the number of significant events BEFORE (preplay) and AFTER (replay) an event (which could be larger than 1)
% PS: Notice that this is not a measure of the percent of episodes with a significant event (which would produce a smaller numbers)
preplayF = nansum(forwardCount(:,start_position))./params.MAX_N_EPISODES;
replayF = nansum(forwardCount(:,above_end_position))./params.MAX_N_EPISODES;
preplayR = nansum(reverseCount(:,start_position))./params.MAX_N_EPISODES;
replayR = nansum(forwardCount(:,above_end_position))./params.MAX_N_EPISODES;

% count number of forward and reverse replays at the choice point
choiceF = nansum(forwardCount(:,choice_position))./params.MAX_N_EPISODES;
choiceR = nansum(reverseCount(:,choice_position))./params.MAX_N_EPISODES;

disp(["preplayF:", preplayF, "preplayR:", preplayR, "replayF", replayF, "replayR", replayR, "choiceF", choiceF, "choiceR", choiceR])

% TODO: see results per episode