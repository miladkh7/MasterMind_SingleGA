function RealGA()
    clc;
    clear;
    close all;

    %% Problem Definition
    global NFE;
    NFE=0;
    secretnumber=randi([0 9],[1 13])
    CostFunction=@(x) GCostFunction(x,secretnumber);     % Cost Function
    nVar=numel(secretnumber);               % Number of Decision Variables
    VarSize=[1 nVar];   % Decision Variables Matrix Size
    VarMin=0;         % Lower Bound of Variables
    VarMax=9;         % Upper Bound of Variables


    %% GA Parameters

    maxIt=200;              % Maximum Number of Iterations
    nPop=80;               % Population Size
    pc=0.8;                 % Crossover Percentage
    nc=2*round(pc*nPop/2);  % Number of Offsprings (Parnets)
    pm=1;                 % Mutation Percentage
    nm=round(pm*nPop);      % Number of Mutants
    mu=0.01;                % Mutation Rate
    beta=8;         % Selection Pressure

    %% Initialization
    BestCost=zeros(maxIt,1);    % Array to Hold Best Cost
    nfe=zeros(maxIt,1);         % Array to Hold Number of Function Evaluations
    people=struct('Position',[],'Cost',[]); 
    pop=repmat(people,nPop,1);
    popc=repmat(people,nc,1);
    popm=repmat(people,nm,1);
    for i=1:nPop
        % Initialize Position
        pop(i).Position=randi([VarMin VarMax],VarSize);
        % Evaluation
        pop(i).Cost=CostFunction(pop(i).Position);
    end

    % Sort Population
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);
    % Store Cost
    WorstCost=pop(end).Cost;

    %% Main Loop

    for it=1:maxIt

        % Calculate Selection Probabilities
        P=exp(-beta*Costs/WorstCost);
        P=P/sum(P);
        % Crossover
        for k=1:nc/2

            % Select Parents Indices
                i1=RouletteWheelSelection(P);
                i2=RouletteWheelSelection(P);
            % Apply Crossover
            [popc(2*k-1,1).Position ,popc(2*k).Position]=Crossover(pop(i1).Position,pop(i2).Position);

            % Evaluate Offsprings
            popc(2*k-1).Cost=CostFunction(popc(2*k-1).Position);
            popc(2*k).Cost=CostFunction(popc(2*k).Position);

        end

        % Select Parent Index
        mutatedPeopled=randsample(nPop,nm);
        % Mutation
        for k=1:nm
            % Apply Mutation
            popm(k).Position=Mutate(pop(mutatedPeopled(k)).Position ,mu,VarMin,VarMax);
            % Evaluate Mutant
            popm(k).Cost=CostFunction(popm(k).Position);
        end

        % Create Merged Population
        pop=[pop;popc;popm];

        % Sort Population
        Costs=[pop.Cost];
        [Costs, SortOrder]=sort(Costs);
        pop=pop(SortOrder);

        % Update Worst Cost
        WorstCost=max(WorstCost,pop(end).Cost);
        % Truncation
        pop=pop(1:nPop);
        Costs=Costs(1:nPop);
        % Store Best Solution Ever Found
        BestSol=pop(1);
        disp(['Iteration ' num2str(it) ': NFE = ' num2str(nfe(it)) 'guess  ' num2str( pop(1).Position) '  , Best Cost = ' num2str(pop(1).Cost)]);
        if(BestSol.Cost==0)
            it
            pop(1).Position
            break;
        end
        % Store Best Cost Ever Found
        BestCost(it)=BestSol.Cost;
        % Store NFE
        nfe(it)=NFE;
        % Show Iteration Information
        
    end

    %% Results


end



%% Functions


function i=RouletteWheelSelection(P)
    r=rand;
    c=cumsum(P);
    i=find(r<=c,1,'first');
end

function y=Mutate(x,mu,VarMin,VarMax)
    nVar=numel(x);
    y=x;
    nmu=ceil(mu*nVar);
    j=randsample(nVar,nmu);
    newnumber=randi([VarMin,VarMax],[numel(j),1]);
    y(j)=newnumber;
end
function [y1 y2]=UniformCrossover(x1,x2)
    alpha=randi([0 1],size(x1));   
    y1=alpha.*x1+(1-alpha).*x2;
    y2=alpha.*x2+(1-alpha).*x1;
end
function [y1 y2]=SinglePointCrossover(x1,x2)
    nVar=numel(x1);
    c=randi([1 nVar-1]); 
    y1=[x1(1:c) x2(c+1:end)];
    y2=[x2(1:c) x1(c+1:end)];
end
function [y1 y2]=DoublePointCrossover(x1,x2)
    nVar=numel(x1);
    cc=randsample(nVar-1,2);
    c1=min(cc);
    c2=max(cc);
    y1=[x1(1:c1) x2(c1+1:c2) x1(c2+1:end)];
    y2=[x2(1:c1) x1(c1+1:c2) x2(c2+1:end)];
end
function [y1 y2]=Crossover(x1,x2)
    pSinglePoint=.2;
    pDoublePoint=.2;
    pUniform=1-pSinglePoint-pDoublePoint;  
    METHOD=RouletteWheelSelection([pSinglePoint pDoublePoint pUniform]);
    switch METHOD
        case 1
            [y1 y2]=SinglePointCrossover(x1,x2);
            
        case 2
            [y1 y2]=DoublePointCrossover(x1,x2);
            
        case 3
            [y1 y2]=UniformCrossover(x1,x2);
    end
end