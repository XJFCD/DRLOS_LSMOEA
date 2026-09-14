function Population = LMOEADS(Problem,Population,RefV,Nw,Ns)

    GuidingSolution = DirectedSampling(Problem,Population,Ns,Nw,RefV);
    Population      = DoubleReproduction(Problem,Population,GuidingSolution,RefV);
end
