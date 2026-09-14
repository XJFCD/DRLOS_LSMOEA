classdef DRLGM_LSMO < ALGORITHM
    % <multi/many> <real> <large/none> <constrained/none>
    % An Adaptive Two-Stage Evolutionary Large-scale multi-objective optimization algorithm
    %------------------------------- Reference --------------------------------
    % Q. Lin, J. Li, S. Liu, et al., An Adaptive Two-Stage Evolutionary
    % Algorithm for Large-Scale Continuous Multi-Objective Optimization, Swarm and
    % Evolutionary Computation, 2023.
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------
    
    % 完整算法，更新了6个参数

    methods
        function main(Algorithm,Problem)
            %% 初始化
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);% 生成均匀参考点，V是每一个点的权重向量，N是点的数量
            Population    = Problem.Initialization(); % 初始化种群
            [z,znad]      = deal(min(Population.objs),max(Population.objs)); % 计算理想点和纳什点

            % 初始化算子参数
            [Ns1,Ns2] = Algorithm.ParameterSet(30,15); % 两阶段采样；Ns1,Ns2垂直和水平采样的数量
            [Nw,Ns] = Algorithm.ParameterSet(10,30); % LMOEADS；Nw聚类数量，Ns每个方向的采样数量
            [Rate,~,~,~] = Algorithm.ParameterSet(0.8,0.4,5,1); % 竞争学习：Rate竞争学习率，控制竞争算子中失败个体向胜利个体学习的强度
            div = Algorithm.ParameterSet(10); %MOPSO；目标空间的划分数量，默认值为10
            [proC,disC] = deal(1,20); %SBX模拟二进制交叉
            [proM,disM] = deal(1,20); %多项式变异
      
            % 初始化DRL参数
            % Data = []; %经验回放缓冲区--------------------------------------------------------
            num_operator = 8;  %算子数量8
            model_built = 0;%是否已构建DQN模型
            count = 0;
            greedy = 0.95;%DQN的超参数（贪婪系数）
            gama = 0.9;%DQN的超参数（折扣因子）
            cnt            = 0; %计数器
            % 预分配Data数组-------------------------------------------------------+
            capacity = 200;
            Data_buffer = cell(capacity, 1);  % 用cell存可变行
            Data_len = 0;
            % 预分配succ_rate数组
            max_gen = 200;  % 最大迭代代数，或者用 Problem.maxFE / Problem.N 估算
            succ_rate = zeros(1, max_gen);

            % 分组
            group_index = build_groups(Problem.D, Problem.M, Problem); % cell array
            G = length(group_index);
            
            % ===== 初始化历史数据（用于奖励归一化）=====
            max_history = 100;  % 最大存储代数
            hist_f_data = zeros(1, max_history);  % 预分配
            hist_d_data = zeros(1, max_history);  % 预分配
            hist_ptr = 1;  % 当前写入位置
            hist_len = 0;  % 实际已存储的数据量（前hist_len个有效）

            % Optimization优化循环-融合DQL
            gen =1;
            while Algorithm.NotTerminated(Population)
                %% DRL算子选择
                progress = Problem.FE / Problem.maxFE;
                
                % 旧状态提取
                average_f = log( mean(sum(Population.objs,2)) + eps );
                average_d = sum( prctile(Population.objs, 90) - prctile(Population.objs, 10) );
                state_old = [average_f, average_d];
                
                flag1 = 0.2;%✅
                if progress <= flag1 %探索阶段，随机选择算子
                    operator = randi(num_operator);
                
                else % 学习阶段：由DQN选择算子
                    
                    if ~model_built 
                        % 构建模型
                        state_dim = length(state_old); % 状态向量维度

                        % 将cell转为矩阵用于训练
                        if Data_len > 0
                            temp_data = vertcat(Data_buffer{1:Data_len});  % 临时转为矩阵
                            use_data = randperm(Data_len, min(30, Data_len)); %从Data中随机选择30条经验
                            tr_x = temp_data(use_data, 1:state_dim+1);%输入特征（状态-动作对）
                            tr_y = temp_data(use_data, state_dim+2:end); %输出数据（奖励-新状态）
                        else
                            continue;  % 没有经验则跳过训练
                        end
                        [tr_xx,ps] = mapminmax(tr_x');tr_xx=tr_xx'; 
                        [tr_yy,qs] = mapminmax(tr_y');tr_yy=tr_yy'; 
                        Params.ps  = ps;Params.qs=qs; %保存归一化参数
                        [net,Params] = trainmodel(tr_xx,tr_yy,Params); %训练DQN模型，返回训练好的网络net和更新后的参数Params
                        model_built = 1; %标记模型已构建
                        fprintf('模型已构建\n');
                        operator = randi(num_operator); %随机选择算子，在模型刚构建后的第一次选择中，仍然使用随机选择
                    
                    else 
                        % 使用模型选择动作，使用ε-贪婪策略
                        if rand > greedy %如果随机概率大于贪婪系数，则采用随机选择算子（探索）
                            operator = randi(num_operator);
                        else %否则，使用DQN选择最优动作（利用）
                            % state_dim = length(state_old); % 状态向量维度
                            ps=Params.ps; %输入数据归一化参数
                            qs=Params.qs; %输出数据归一化参数

                            test_inputs = [repmat(state_old, num_operator, 1), (1:num_operator)']; %完全向量化：一次构建所有测试输入 状态-动作
                            test_inputs_normalized = mapminmax('apply', test_inputs', ps)'; %批量归一化
                            rewards_normalized = testNet(test_inputs_normalized, net, Params); %DQN批量预测奖励
                            rewards = mapminmax('reverse', rewards_normalized', qs)'; % 批量反归一化
                            % 选择最优动作
                            [~,operator] = max(rewards(:, 1));% 选择奖励最大的动作（算子）
                        end
                    end
                end

                %% DRL根据选择的算子进行进化操作
                cnt =cnt+1; %记录迭代次数

                flag2=0.3; %✅

                if  progress<=flag2 %

                    %进化前期阶段，全维度使用算子
                    switch operator
                        case 1 % GA
                            valOffspring = OperatorGAhalf(Problem, Population(randi(length(Population), 1, Problem.N)));
                        case 2 % DE
                            valOffspring = OperatorDE(Problem, Population, Population(randi(ceil(length(Population)), 1, Problem.N)), Population(randi(ceil(length(Population)), 1, Problem.N)));
                        case 3 % LMOEADS采样
                            valOffspring = Operator_LMOEADS(Problem, Population, V, Nw, Ns);
                        case 4 % 竞争学习
                            valOffspring = CompetitiveLearning(Problem, Population, z, znad, V, Rate); 
                        case 5 % 两阶段采样 
                            valOffspring = TwoStageSampling(Problem, Population, V, Ns1, Ns2);
                        case 6 % MOPSO
                            valOffspring = Operator_MOPSO(Problem, Population, div);
                        case 7 % SBX
                            valOffspring = Operator_SBX(Population, Problem, proC, disC);
                        case 8 % 多项式变异
                            valOffspring = Operator_PM(Population, Problem, proM, disM);
                    end
                    
                    %环境选择
                    SelectionPool = [Population, valOffspring];
                    [Population, ~, ~] = EnvironmentalSelection(SelectionPool, length(Population), 2); 
                
                else 
                    %进化中/后期，局部应用算子
                    alpha = 0.3;%✅
                                        
                    switch operator
                        case 1 % GA √
                                % valOffspring = OperatorGAhalf(Problem, Population(randi(length(Population), 1, Problem.N))); 
                                OffDec  = OperatorGAhalfonly(Problem, Population(randi(length(Population), 1, Problem.N)));
                                No = size(OffDec,1);
                                ParentIdx = randi(length(Population), No, 1);
                                ParentDecRef = reshape([Population(ParentIdx).dec], Problem.D, [])';
                                
                                % mask 操作
                                mask = false(1, Problem.D);
                                Ksel  = max(1, round(alpha * G));
                                selected_groups = randperm(G, Ksel);
                                % 向量化：一次性收集所有索引
                                all_indices = [group_index{selected_groups}];
                                mask(all_indices) = true;
                                OffDec(:, ~mask) = ParentDecRef(:, ~mask);
                                
                                % 再评估
                                valOffspring = Problem.Evaluation(OffDec);
                            case 2 % DE √
                                % valOffspring = OperatorDE(Problem, Population, Population(randi(ceil(length(Population)), 1, Problem.N)), Population(randi(ceil(length(Population)), 1, Problem.N)));
                                OffDec   = OperatorDEonly(Problem, Population, Population(randi(ceil(length(Population)), 1, Problem.N)), Population(randi(ceil(length(Population)), 1, Problem.N)));
                                ParentDec   = reshape([Population.dec], Problem.D, [])';
                                mask = false(1, Problem.D);
                                Ksel  = max(1, round(alpha * G));
                                selected_groups = randperm(G, Ksel);

                                all_indices = [group_index{selected_groups}];
                                mask(all_indices) = true;
                                OffDec(:, ~mask) = ParentDec(:, ~mask);
                                valOffspring = Problem.Evaluation(OffDec);
                            case 3 % LMOEADS采样，评估次数消耗1000
                                valOffspring = Operator_LMOEADS(Problem, Population, V, Nw, Ns);
                            case 4 % 竞争学习 √
                                valOffspring = CompetitiveLearning(Problem, Population, z, znad, V, Rate);
                            case 5 % 两阶段采样，评估次数消耗1000 ~2000
                                valOffspring = TwoStageSampling(Problem, Population, V, Ns1, Ns2);
                            case 6 % MOPSO √
                                valOffspring = Operator_MOPSO(Problem, Population, div);
                            case 7 % SBX √
                                OffDec = Operator_SBXonly(Population, Problem, proC, disC);
                                ParentDec   = reshape([Population.dec], Problem.D, [])';
                                
                                mask = false(1, Problem.D);
                                Ksel  = max(1, round(alpha * G));
                                selected_groups = randperm(G, Ksel);

                                all_indices = [group_index{selected_groups}];
                                mask(all_indices) = true;
                                OffDec(:, ~mask) = ParentDec(:, ~mask);
                                valOffspring = Problem.Evaluation(OffDec);
                            case 8 % 多项式变异PM √
                                OffDec = Operator_PMonly(Population, Problem, proM, disM);
                                ParentDec   = reshape([Population.dec], Problem.D, [])';
                                
                                mask = false(1, Problem.D);
                                Ksel  = max(1, round(alpha * G));
                                selected_groups = randperm(G, Ksel);

                                all_indices = [group_index{selected_groups}];
                                mask(all_indices) = true;
                                OffDec(:, ~mask) = ParentDec(:, ~mask);
                                valOffspring = Problem.Evaluation(OffDec);
                    end
                    
                    % 计算成功率，评估子代质量，计算公式：(父代被选中的比例) - (子代被选中的比例)/2
                    [~,~,Next] = EnvironmentalSelection( [Population,valOffspring],length(Population),2);
                    succ_rate(1,cnt) =  (sum(Next(1:length(Population)))/100) - (sum(Next(length(Population)+1:end))/50);
                    %智能迁移决策
                    if   succ_rate(1,cnt) >0 % 如果子代质量好
                        currentPopSize = length(Population);
                        % 增加多样性：引入更多探索性解
                        % additional_solutions = Population(rand_number(1:ceil(Problem.N/4)));
                        rand_number = randperm(currentPopSize);
                        selectCount = min(ceil(currentPopSize/4), currentPopSize);
                        additional_solutions = Population(rand_number(1:selectCount));
                        [Population,~,~] = EnvironmentalSelection( [Population,valOffspring,additional_solutions],length(Population),2);
                    else % 如果子代质量差
                        % 加强利用：主要保留高质量子代 
                        [Population,~,~] = EnvironmentalSelection( [Population,valOffspring],length(Population),2);
                    end

                end
                

                %% 新状态提取
                average_f1 = log( mean(sum(Population.objs,2)) + eps );
                average_d1 = sum( prctile(Population.objs, 90) - prctile(Population.objs, 10) );
                state_new = [average_f1, average_d1];
                
                %% ===== 统一尺度的奖励函数（Z-score 标准化）=====
                delta_f = average_f - average_f1;
                delta_d = average_d1 - average_d;

                % 更新历史数据 - 环形缓冲区写入
                hist_f_data(hist_ptr) = delta_f;
                hist_d_data(hist_ptr) = delta_d;
                
                % 更新指针和长度
                hist_ptr = hist_ptr + 1;
                hist_len = min(hist_len + 1, max_history);
                if hist_ptr > max_history
                    hist_ptr = 1;  % 循环到开头
                end
                
                % 计算历史标准差并归一化
                if hist_len >= 20
                    % 提取有效数据（前hist_len个，因为环形缓冲区从1开始顺序写入）
                    valid_f_data = hist_f_data(1:hist_len);
                    valid_d_data = hist_d_data(1:hist_len);
                    
                    f_std = std(valid_f_data);
                    d_std = std(valid_d_data);
                    
                    if f_std < 1e-10, f_std = 1e-10; end
                    if d_std < 1e-10, d_std = 1e-10; end
                    
                    conv_reward = tanh(delta_f / f_std);
                    div_reward = tanh(delta_d / d_std);
                else
                    % 初期：使用固定缩放
                    conv_reward = tanh(delta_f * 1e5);
                    div_reward = tanh(delta_d / 5000);
                end
                
                w_conv = 0.7;
                w_div = 0.3;
                reward1 = w_conv * conv_reward + w_div * div_reward;
                % reward1 = (average_f - average_f1) + 0.1 * (average_d1 - average_d);%计算真实奖励

                %% 更新经验数据
                current_record = [state_old, operator, reward1, state_new]; %组织新经验数据

                % 使用预分配的cell数组存储
                Data_len = Data_len + 1;
                if Data_len > capacity
                    % FIFO：移除最早的一条,先进先出，整体左移
                    Data_buffer(1) = [];
                    Data_buffer{capacity} = current_record;
                    Data_len = capacity;
                else
                    Data_buffer{Data_len} = current_record;
                end
                
                %% 更新DQN模型
                if model_built
                    count = count + 1;
                    if count > 20 %frequent✅
                        % update model here
                        qs=Params.qs;
                        state_dim = length(state_old); % 状态向量维度

                        % ===== 从cell数组中提取训练数据 =====
                        temp_data = vertcat(Data_buffer{1:Data_len});% 将cell转为矩阵（只在需要时转换）
                        use_data = randperm(Data_len, min(20, Data_len));%batchsize✅
                        tr_x = temp_data(use_data,1:state_dim+1); %输入特征
                        [tr_xx,ps] = mapminmax(tr_x');tr_xx=tr_xx'; %输入特征进行归一化处理
                        reward1 = testNet(tr_xx,net,Params); %使用当前网络预测奖励
                        reward1=mapminmax('reverse',reward1',qs);reward1=reward1'; %反归一化得到实际尺度的奖励
                        succ = reward1(:,1);%提取预测的奖励
                        tr_yy = temp_data(use_data,state_dim+2)+gama*max(succ);%计算奖励=真实奖励+折扣因子*预测的下一个状态的最大奖励。max(succ) 是同一批数据中所有状态-动作对的奖励的最大值，近似下一个状态的最大奖励（假设相邻代的状态变化不大）
                        [tr_yy,qs] = mapminmax(tr_yy');tr_yy=tr_yy';%目标值归一化
                        Params.ps  = ps;Params.qs=qs;
                        net = updatemodel(tr_xx,tr_yy,Params,net);%更新网络权重
                        count = 0;
                        fprintf('模型已更新\n');
                    end
                end
                gen = gen+1;
            end
        end
    end
end


% 变量分组
function group_index = build_groups(D, ~, ~)

    G = 6; %✅
    gs = floor(D / G);
    group_index = cell(G,1);
    for g = 1:G
        if g < G
            group_index{g} = (g-1)*gs+1 : g*gs;
        else
            group_index{g} = (g-1)*gs+1 : D;
        end
    end
    %end
end
