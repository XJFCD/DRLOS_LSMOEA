function Offspring = Operator_SBX(Population, Problem, proC, disC)
% Migrated from the PlatEMO OperatorGA module for convenience
% This function is written by Ian Meyer Kropp
    if isa(Population(1), 'SOLUTION')
        evaluated = true;
        Parent = Population.decs;
    else
        evaluated = false;
        Parent = Population;
    end

    N = floor(length(Population)/2);
    Parent1 = Parent(1:N, :);
    Parent2 = Parent(N+1:2*N, :);
    [N,D]   = size(Parent1);
    beta = zeros(N,D);
    mu   = rand(N,D);

    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;

    Offspring1 = (Parent1 + Parent2)/2 + beta .* (Parent1 - Parent2)/2;
    Offspring2 = (Parent1 + Parent2)/2 - beta .* (Parent1 - Parent2)/2;
    OffDec = [Offspring1; Offspring2];

    Lower = repmat(Problem.lower, size(OffDec,1), 1);
    Upper = repmat(Problem.upper, size(OffDec,1), 1);
    OffDec = max(min(OffDec, Upper), Lower);

    if evaluated
        Offspring = Problem.Evaluation(OffDec);
    else
        Offspring = OffDec;
    end
end