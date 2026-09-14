function [OffDec, OffVel] = CompetitiveLearningonly(Problem, Population, z, znad, V, Rate)
    % 竞争学习
    [P1, P2, A, B] = classification1(Population, z, znad, V);
    [~, ~, L1, L2] = classification1(P1, z, znad, V);
    [~, ~, L3, L4] = classification1(P2, z, znad, V);
    
    Rank = randperm(length(Population), floor(length(Population)/2)*2);
    Loser = Rank(1:end/2);
    Winner = Rank(end/2+1:end);
    
    a1 = A(L1); a2 = B(L3);
    b1 = A(L2); b2 = B(L4);
    
    loser = zeros(1, size(Loser, 2));
    winner = zeros(1, size(Winner, 2));
    
    for i = 1:length(Population)/4
        loser(Loser == a1(i)) = 1;
        loser(Loser == b1(i)) = 2;
        loser(Loser == a2(i)) = 3;
        loser(Loser == b2(i)) = 4;
    end
    
    for i = 1:length(Population)/4
        winner(Winner == a1(i)) = 1;
        winner(Winner == b1(i)) = 2;
        winner(Winner == a2(i)) = 3;
        winner(Winner == b2(i)) = 4;
    end
    
    Change = loser < winner;
    Exam = loser == winner;
    Change1 = [];
    
    Fitness = CalFitness(Population.objs);
    for i = 1:length(Exam)
        if (Exam(i) == true && (Fitness(Loser(i))) > Fitness(Winner(i)))
            Change1 = [Change1, i];
        end
    end
    
    Change(Change1) = 1;
    Temp = Winner(Change);
    Winner(Change) = Loser(Change);
    Loser(Change) = Temp;
    
    [OffDec, OffVel] = Operator_LMOCSO(Problem, Population(Loser), Population(Winner), Rate);
    %Offspring = Problem.Evaluation(OffDec, OffVel);

end