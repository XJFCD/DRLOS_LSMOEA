% classdef MOPSO < ALGORITHM
% % <2002> <multi> <real/integer>
% % Multi-objective particle swarm optimization
% % div --- 10 --- The number of divisions in each objective
% 
% %------------------------------- Reference --------------------------------
% % C. A. Coello Coello and M. S. Lechuga. MOPSO: A proposal for multiple
% % objective particle swarm optimization. Proceedings of the IEEE Congress
% % on Evolutionary Computation, 2002, 1051-1056.
% %------------------------------- Copyright --------------------------------
% % Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% % research purposes. All publications which use this platform or any code
% % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% % for evolutionary multi-objective optimization [educational forum], IEEE
% % Computational Intelligence Magazine, 2017, 12(4): 73-87".
% %--------------------------------------------------------------------------
% 
% 	methods
function Population = Operator_MOPSO(Problem, Population, div)
    %% Parameter setting
    %div = Algorithm.ParameterSet(10); %目标空间的划分数量，默认值为10

    %% Generate random population
    %Population = Problem.Initialization(); % 初始化种群，Population：当前粒子群
    Archive    = UpdateArchive_MOPSO(Population,Problem.N,div); % 初始化外部存档，Archive：外部档案，存储非支配解
    Pbest      = Population; % 个体历史最优位置初始化为初始种群，Pbest：每个粒子的个体历史最优位置

    %% Optimization
    %while Algorithm.NotTerminated(Archive)
    REP        = REPSelection(Archive.objs,Problem.N,div); % 从外部档案中选择全局引导粒子（gbest），REP包含被选为引导者的粒子索引
    Population = OperatorPSO(Problem,Population,Pbest,Archive(REP)); %更新粒子位置和速度，每个粒子向自己的pbest和全局的gbest学习
        %Archive    = UpdateArchive([Archive,Population],Problem.N,div); %更新外部存档
        %Pbest      = UpdatePbest(Pbest,Population); %更新个体历史最优
    %end
end
%     end
% end