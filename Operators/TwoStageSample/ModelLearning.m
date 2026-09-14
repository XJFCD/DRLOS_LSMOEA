function mlp = ModelLearning(Population)
% Training the MLP model
% 输入：当前种群(Population)
% 输出：训练好的MLP模型(mlp)
% 作用：学习从较差个体(Loser)到较优个体(Winner)的映射关系，用于后续的采样指导

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Prepare the training data
    Fitness = calFitness(Population.objs); %计算种群中每个个体的适应度
    if length(Population) >= 2
        Rank = randperm(length(Population),floor(length(Population)/2)*2); %从种群中随机选取一半的个体(偶数个)，并返回它们的索引。*2：将结果乘以2，确保最终选取的个体数是 偶数
    else
        Rank = [1,1];
    end
    % 随机将种群分成"失败者"(Loser)和"胜利者"(Winner)两组
    Loser  = Rank(1:end/2);
    Winner = Rank(end/2+1:end);
    % 调整分组：确保Winner组的适应度优于Loser组，如果Loser适应度更好，则交换两者位置
    Change = Fitness(Loser) >= Fitness(Winner); % 判断败者的适应度是否≥胜者，返回一个逻辑向量（Change）           
    Temp   = Winner(Change); % 临时存储胜者中需要交换的个体           
    Winner(Change) = Loser(Change); %将Loser中满足Change条件的个体赋值给Winner的对应位置            
    Loser(Change)  = Temp; %将暂存的原Winner个体（Temp）赋值给Loser的对应位置
    % 提取失败者和胜利者的决策变量
    LoserDec  = Population(Loser).decs;
    WinnerDec = Population(Winner).decs;
    % [P,LabelSet]= classification(Population);
%     LoserDec  = P.decs;
%     WinnerDec = LabelSet.decs;
    num_D     = size(LoserDec,2); %获取决策空间的维度
    
    %% Train MLP
    epoch = 20;
    mlp = MMLP(num_D,num_D,1,20,0.5,0.1); %创建MLP模型：输入层和输出层大小=决策变量维度(num_D)，隐藏层数量=1，每个隐藏层中的神经元数量=20，学习率 = 0.5，动量系数 = 0.1
    for i = 1:epoch
        mlp.train(LoserDec, WinnerDec); %训练20个epoch，学习从Loser到Winner的映射
    end 
end

function Fitness = calFitness(PopObj)
% Calculate the fitness by shift-based density（基于位移的密度估计）
% 输入：种群目标值矩阵PopObj (N×M，N个个体，M个目标)
% 输出：适应度向量Fitness (N×1)
% 作用：通过计算每个个体到其"位移后"邻居的最小距离来衡量解的质量
% 原理：对于个体A，将其他个体向A方向移动，若移动后A的距离密度小，则形成拥挤区域，A将被删除

    N      = size(PopObj,1); %种群个数
    fmax   = max(PopObj,[],1); %返回每个目标的最大值
    fmin   = min(PopObj,[],1); %返回每个目标的最小值
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1); % 将目标值归一化到[0,1]
    Dis    = inf(N); % 初始化距离矩阵（N*N，元素均为inf）
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1)); %比较个体i和其他所有个体在每一个目标上的值，返回较大的
        for j = [1:i-1,i+1:N] %遍历除i以外的其他所有个体
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:)); % 计算i与（向i方向）位移后的j的欧氏距离（只计算j中比i更大的目标值）
        end
    end
    Fitness = min(Dis,[],2); %每个个体的适应度是其与最近位移后邻居的距离，值越小表示该个体所在区域越拥挤
end