function Offspring = Operator_PM(Population, Problem, proM, disM)
    
    if nargin < 3 || isempty(proM)
        proM = 1;
    end
    if nargin < 4 || isempty(disM)
        disM = 20;
    end

    if isa(Population(1), 'SOLUTION')
        evaluated = true;
        Parent = Population.decs;
    else
        evaluated = false;
        Parent = Population;
    end

    [N,D]   = size(Parent);
    OffDec = Parent; 
    
    %% Polynomial mutation
    Lower = repmat(Problem.lower, N,1);
    Upper = repmat(Problem.upper, N,1);
    Site  = rand(N,D) < proM/D; 
    mu    = rand(N,D);

    temp  = Site & mu<=0.5;
    OffDec       = min(max(OffDec,Lower),Upper);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

    OffDec = max(min(OffDec, Upper), Lower);
   
    if evaluated
        Offspring = Problem.Evaluation(OffDec);
    else
        Offspring = OffDec;
    end
end