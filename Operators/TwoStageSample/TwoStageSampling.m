function Offspring = TwoStageSampling(Problem, Population, V, Ns1, Ns2)

    [~, sizepop] = size(Population);
    sizepop1 = sizepop/2;
    subp1 = []; subp3 = [];
    
    for i = 1:sizepop1
        subp1 = [subp1; Population(1, randi([1, sizepop], 1))];
        subp3 = [subp3; Population(1, randi([1, sizepop], 1))];
    end

    mlp = ModelLearning(subp1);
    [subPop1new, ~] = mlp.forward(subp1.decs);
    subPop1new = Problem.Evaluation(subPop1new);
    subp1fn = subp1((NDSort(subPop1new.objs, 1)==1));
    subPop1 = DirectedSampling2(subp1fn, Ns2, V, Problem);

    mlp = ModelLearning(subp3);
    [subPop3new, ~] = mlp.forward(subp3.decs);
    subPop3new = sort(subPop3new, 2);
    subp3 = Problem.Evaluation(sort(subp3.decs));
    subPop3fn = subp3((NDSort(subp3.objs, 1)==1));
    subPop3new = Problem.Evaluation(subPop3new);
    subPop3newfn = subPop3new((NDSort(subPop3new.objs, 1)==1));
    
    [~, nnew3] = size(subPop3newfn);
    [~, n3] = size(subPop3fn);
    
    if nnew3 > n3
        subPop3newfn = subPop3newfn(:, 1:n3);
    else
        subPop3fn = subPop3fn(:, 1:nnew3);
    end
    
    subPop3 = DirectedSampling3(subPop3newfn, subPop3fn, Ns1, V, Problem);
    Offspring = [subPop1, subPop3];
end