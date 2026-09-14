function Population = Operator_MOPSO(Problem, Population, div)

    Archive    = UpdateArchive_MOPSO(Population,Problem.N,div); 
    Pbest      = Population; 

    % Optimization
    REP        = REPSelection(Archive.objs,Problem.N,div); 
    Population = OperatorPSO(Problem,Population,Pbest,Archive(REP)); 
end
