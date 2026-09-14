classdef DRLOSEMCMO_NoCon < ALGORITHM
% <2024> <multi> <real/integer/label/binary/permutation> <constrained>
% EMCMO with deep reinforcement learning-assisted operator selection

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, L. Wang, and Y. Jin. Constrained multi-objective
% optimization with deep reinforcement learning assisted operator
% selection. IEEE/CAA Journal of Automatica Sinica, 2024, 11(4): 919-931.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    methods                                                                                                                                                                                                                                                                                                                                                                                     
        function main(Algorithm,Problem)
            % 初始化种群
            Population{1}  = Problem.Initialization(); %主种群
            Population{2}  = Problem.Initialization(); %辅助种群
            % 计算两个种群的适应度值
            Fitness{1}     = CalFitness(Population{1}.objs); %,Population{1}.cons含约束处理，传入objs和cons
            Fitness{2}     = CalFitness(Population{2}.objs); %无约束处理
            transfer_state = 0; %是否开启迁移机制（0/1）
            cnt            = 0; %计数器
            
            %% For DQL
            Data = []; %经验回放缓冲区，数据结构：t = (s, op, r, s′) = (con, fea, div, op, r, con′, fea′, div′)
            num_operator = 2; %算子数量（2个：GA 和 DE）
            model_built = 0; %是否已构建DQN模型
            count = 0;
            greedy = 0.95; %DQN的超参数（贪婪系数）
            gama = 0.9; %DQN的超参数（折扣因子）
            
            %% Optimization 主循环
            while Algorithm.NotTerminated(Population{1})
                gen = ceil(Problem.FE/(2*Problem.N)); %计算当前进化的代数
                
                % old state 状态提取（用于DQN）
                average_f = sum(sum(Population{1}.objs,2))/length(Population{1}); %目标函数平均值
                % cv = overall_cv(Population{1}.cons);
                % average_cv = sum(cv)/length(Population{1});%约束违反平均值
                f_max = max(Population{1}.objs,[],1);
                f_min = min(Population{1}.objs,[],1);
                average_d = sum(f_max-f_min); %目标空间分布广度
                
                if gen <= 200 %若当前代数小于200代，则是探索阶段
                    %% Exploring stage, choose operator randomly探索阶段：随机选择算子
                    operator = randi(num_operator);
                else % 当前代数大于200代，则是学习阶段
                    %% Learning stage, choose by Deep Q-net学习阶段：由DQN选择算子
                    % choose an action based on the trained net
                    if ~model_built % 如果还没有构建DQN模型
                        % 构建模型：从经验回放缓冲区Data中采样数据，训练一个DQN，用于后续的算子选择
                        use_data = randperm(length(Data),200); %从Data中随机选择200条经验
                        tr_x = Data(use_data,1:3); %输入特征为数据的前4列（状态-动作对）
                        [tr_xx,ps] = mapminmax(tr_x');tr_xx=tr_xx'; %输入数据归一化
                        tr_y = Data(use_data,4:6); %输出数据（奖励-新状态），Data的后4列
                        [tr_yy,qs] = mapminmax(tr_y');tr_yy=tr_yy'; %输出数据归一化
                        Params.ps  = ps;Params.qs=qs; %保存归一化参数
                        [net,Params] = trainmodel(tr_xx,tr_yy,Params); %训练DQN模型，返回训练好的网络net和更新后的参数Params
                        model_built = 1; %标记模型已构建
                        operator = randi(num_operator); %随机选择算子，在模型刚构建后的第一次选择中，仍然使用随机选择
                    else %如果已构建DQN模型
                        % 使用模型选择动作，使用ε-贪婪策略
                        if rand > greedy %如果随机概率大于贪婪系数，则采用随机选择算子（探索）
                            operator = randi(num_operator);
                        else %否则，使用DQN选择最优动作（利用）
                            %构建测试输入
                            test_x1 = [average_f,average_d,1]; % 状态+动作1average_cv,
                            test_x2 = [average_f,average_d,2]; % 状态+动作2average_cv,
                            ps=Params.ps; %输入数据归一化参数
                            qs=Params.qs; %输出数据归一化参数
                            %数据归一化
                            x1=mapminmax('apply',test_x1',ps);x1=x1';
                            x2=mapminmax('apply',test_x2',ps);x2=x2';
                            %DQN预测奖励
                            reward = testNet(x1,net,Params); %预测选择动作1的奖励
                            reward=mapminmax('reverse',reward',qs);reward=reward'; %反归一化，将网络输出反归一化到原始尺度
                            succ2 = testNet(x2,net,Params); %预测选择动作2的奖励
                            succ2=mapminmax('reverse',succ2',qs);succ2=succ2'; %反归一化
                            % 选择最优动作
                            succ = [reward;succ2];% 组合两个动作的奖励
                            [~,operator] = max(succ(:,1));% 选择奖励最大的动作（算子）
                        end
                    end
                end
                
                cnt =cnt+1; %记录迭代次数

                if   transfer_state == 0 %进化前期阶段：两个种群相对独立地进化
                %在算法运行前期（前20%的评估次数内），两个种群采用相对独立的进化策略，但已经开始交换子代个体。
                    %生成子代个体（随机选择）
                    for i = 1: 2 %遍历两个种群（两个种群都使用相同的算子）
                        if operator == 1 %选择算子1，采用遗传算法生成子子代
                            valOffspring{i} = OperatorGAhalf(Problem,Population{i}(randi(Problem.N,1,Problem.N)));
                        else %选择算子2，采用差分进化
                            valOffspring{i} = OperatorDE(Problem,Population{i},Population{i}(randi(ceil(Problem.N),1,Problem.N)),Population{i}(randi(ceil(Problem.N),1,Problem.N)));
                        end
                    end
                    %环境选择：两个种群开始交换子代个体，为后续迁移做准备
                    for i = 1:2 %遍历两个种群
                        if i == 1 %种群1的选择池：种群1 + 种群1的子代 + 种群2的子代
                            [Population{i},Fitness{i},~] = EnvironmentalSelection( [Population{1},valOffspring{1},valOffspring{2}],Problem.N,i);
                        else %种群2的选择池：种群2 + 种群2的子代 + 种群1的子代
                            [Population{i},Fitness{i},~] = EnvironmentalSelection( [Population{2},valOffspring{2},valOffspring{1}],Problem.N,i);
                        end
                    end
                    % 阶段转换条件
                    if Problem.FE/Problem.maxFE >=0.2 %当函数评估次数达到总次数的20%时，切换到迁移阶段
                        transfer_state = 1;
                    end
                    
                else %进化中/后期，采用成功率评估和智能迁移机制
                    % 生成子代个体（锦标赛选择）
                    for i = 1: 2 %遍历两个种群
                        if operator == 1 %如果是GA算子
                            MatingPool = TournamentSelection(2,Problem.N,Fitness{i}); %锦标赛选择组成交配池
                            valOffspring{i} = OperatorGAhalf(Problem,Population{i}(MatingPool)); %遗传算法生成子代
                        else %如果是DE算子
                            MatingPool = TournamentSelection(2,2*Problem.N,Fitness{i}); %锦标赛选择交配池（规模更大）
                            valOffspring{i} = OperatorDE(Problem,Population{i},Population{i}(MatingPool(1:end/2)),Population{i}(MatingPool(end/2+1:end))); %差分进化生成子代
                        end
                    end
                    
                    % 计算成功率，计算公式：(父代被选中的比例) - (子代被选中的比例)/2
                    % 反映了子代的质量：正值表示子代质量好，负值表示子代质量差
                    % Next：布尔向量，表示哪些个体被选中到下一代
                    [~,~,Next] = EnvironmentalSelection( [Population{2},valOffspring{2}],Problem.N,1);
                    succ_rate(1,cnt) =  (sum(Next(1:Problem.N))/100) - (sum(Next(Problem.N+1:end))/50);
                    [~,~,Next] = EnvironmentalSelection( [Population{1},valOffspring{1}],Problem.N,2);
                    succ_rate(2,cnt) =  (sum(Next(1:Problem.N))/100) - (sum(Next(Problem.N+1:end))/50);
                    
                    %智能迁移决策
                    for i = 1:2
                        if   succ_rate(i,cnt) >0 % 如果子代质量好
                            rand_number = randperm(Problem.N);
                            % 选择池 = [当前种群, 当前子代，另一种群的随机50%个体]
                            % 引入另一种群的个体来增加多样性。因为当前搜索效果好，可以承受更多的多样性
                            [Population{i},Fitness{i},~] = EnvironmentalSelection( [Population{i},valOffspring{i},Population{2/i}(rand_number(1:Problem.N/2))],Problem.N,i);
                        else % 如果子代质量差
                            % 选择池 = [当前种群, 当前子代, 另一种群的子代]
                            % 引入另一种群的高质量子代。因为当前搜索效果差，需要借助另一种群的成功经验
                            [Population{i},Fitness{i},~] = EnvironmentalSelection( [Population{i},valOffspring{i},valOffspring{2/i}],Problem.N,i);
                        end
                    end
                end
                
                %% new stage 状态提取
                average_f1 = sum(sum(Population{1}.objs,2))/length(Population{1}); %目标函数平均值
                % cv1 = overall_cv(Population{1}.cons);
                % average_cv1 = sum(cv1)/length(Population{1});%约束违反平均值
                f_max1 = max(Population{1}.objs,[],1);
                f_min1 = min(Population{1}.objs,[],1);
                average_d1 = sum(f_max1-f_min1);%目标空间分布广度
                
                %% Update experience replay更新经验数据
                reward = (average_f1 + average_d1) - (average_f + average_d); %计算真实奖励
                current_record = [average_f average_d operator reward,average_f1 average_d1]; %组织新经验数据
                Data = [Data;current_record];
                if size(Data,1) > 500 %获取Data矩阵的行数（即存储的经验条数）检查当前经验数量是否超过500条
                    Data(end,:) = []; %先进先出（FIFO） 策略：最先存入的经验最先被删除，Data(end,:) 表示矩阵的最后一行（最旧的经验）
                end
                
                %% Update Q-net
                % update net model every 50 generations每50代更新一次DQN模型
                if model_built
                    count = count + 1;
                    if count > 50
                        % update model here
                        qs=Params.qs;
                        use_data = randperm(length(Data),200);% 训练数据：从经验数据库随机选择200条经验
                        tr_x = Data(use_data,1:3); %输入特征
                        [tr_xx,ps] = mapminmax(tr_x');tr_xx=tr_xx'; %输入特征进行归一化处理
                        reward = testNet(tr_xx,net,Params); %使用当前网络预测奖励
                        reward=mapminmax('reverse',reward',qs);reward=reward'; %反归一化得到实际尺度的奖励
                        succ = reward(:,1);%提取预测的奖励
                        tr_yy = Data(use_data,4)+gama*max(succ);%计算奖励=真实奖励+折扣因子*预测的下一个状态的最大奖励。max(succ) 是同一批数据中所有状态-动作对的奖励的最大值，近似下一个状态的最大奖励（假设相邻代的状态变化不大）
                        [tr_yy,qs] = mapminmax(tr_yy');tr_yy=tr_yy';%目标值归一化
                        Params.ps  = ps;Params.qs=qs;
                        net = updatemodel(tr_xx,tr_yy,Params,net);%更新网络权重
                        count = 0;
                    end
                end
            end
        end
    end
end

% function result = overall_cv(cv)
% % The overall constraint violation
% 
%     cv(cv <= 0) = 0;
%     cv = abs(cv);
%     result = sum(cv,2);
% end
function Fitness = CalFitness(PopObj,PopCon)
% Calculate the fitness of each solution


    N = size(PopObj,1);
    if nargin == 1
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end