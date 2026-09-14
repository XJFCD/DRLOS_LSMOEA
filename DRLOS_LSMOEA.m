classdef DRLOS_LSMOEA < ALGORITHM
    % <multi/many> <real> <large> 
    % A Deep Reinforcement Learning-Assisted Operator Selection for Large-Scale Multi-Objective Optimization
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

    methods
        function main(Algorithm,Problem)

            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Population    = Problem.Initialization(); 
            [z,znad]      = deal(min(Population.objs),max(Population.objs)); 
            % Operator parameters
            [Ns1,Ns2] = Algorithm.ParameterSet(30,15);
            [Nw,Ns] = Algorithm.ParameterSet(10,30); 
            [Rate,~,~,~] = Algorithm.ParameterSet(0.8,0.4,5,1);
            div = Algorithm.ParameterSet(10); %MOPSO；
            [proC,disC] = deal(1,20); 
            [proM,disM] = deal(1,20); 
            % DRL parameters
            num_operator = 8;  
            model_built = 0;
            count = 0;
            greedy = 0.95;
            gama = 0.9;
            cnt= 0; 
            % Pre-allocate the Data array
            capacity = 200;
            Data_buffer = cell(capacity, 1); 
            Data_len = 0;
            max_gen = 200; 
            succ_rate = zeros(1, max_gen);

            % Group
            group_index = build_groups(Problem.D); % cell array
            G = length(group_index);
            
            % Initialize historical data
            max_history = 100; 
            hist_f_data = zeros(1, max_history);  
            hist_d_data = zeros(1, max_history); 
            hist_ptr = 1;  
            hist_len = 0;  

            % Optimization
            gen =1;
            while Algorithm.NotTerminated(Population)

                progress = Problem.FE / Problem.maxFE;
                
                average_f = log( mean(sum(Population.objs,2)) + eps );
                average_d = sum( prctile(Population.objs, 90) - prctile(Population.objs, 10) );
                state_old = [average_f, average_d];
                
                flag1 = 0.2;
                if progress <= flag1 
                    operator = randi(num_operator);
                else                  
                    if ~model_built 
                        state_dim = length(state_old);
                        if Data_len > 0
                            temp_data = vertcat(Data_buffer{1:Data_len}); 
                            use_data = randperm(Data_len, min(30, Data_len)); 
                            tr_x = temp_data(use_data, 1:state_dim+1);
                            tr_y = temp_data(use_data, state_dim+2:end); 
                        else
                            continue; 
                        end
                        [tr_xx,ps] = mapminmax(tr_x');tr_xx=tr_xx'; 
                        [tr_yy,qs] = mapminmax(tr_y');tr_yy=tr_yy'; 
                        Params.ps  = ps;Params.qs=qs; 
                        [net,Params] = trainmodel(tr_xx,tr_yy,Params); 
                        model_built = 1; 
                        operator = randi(num_operator); 
                    else 
                        if rand > greedy 
                            operator = randi(num_operator);
                        else 
                            ps=Params.ps; 
                            qs=Params.qs; 
                            test_inputs = [repmat(state_old, num_operator, 1), (1:num_operator)']; 
                            test_inputs_normalized = mapminmax('apply', test_inputs', ps)'; 
                            rewards_normalized = testNet(test_inputs_normalized, net, Params); 
                            rewards = mapminmax('reverse', rewards_normalized', qs)';
                            [~,operator] = max(rewards(:, 1));
                        end
                    end
                end

                cnt =cnt+1; 

                flag2=0.3;
                if  progress<=flag2 
                    switch operator
                        case 1 % GA
                            valOffspring = OperatorGAhalf(Problem, Population(randi(length(Population), 1, Problem.N)));
                        case 2 % DE
                            valOffspring = OperatorDE(Problem, Population, Population(randi(ceil(length(Population)), 1, Problem.N)), Population(randi(ceil(length(Population)), 1, Problem.N)));
                        case 3 % LMOEADS
                            valOffspring = Operator_LMOEADS(Problem, Population, V, Nw, Ns);
                        case 4 % Competitive Learning
                            valOffspring = CompetitiveLearning(Problem, Population, z, znad, V, Rate); 
                        case 5 % Two-stage Sampling 
                            valOffspring = TwoStageSampling(Problem, Population, V, Ns1, Ns2);
                        case 6 % MOPSO
                            valOffspring = Operator_MOPSO(Problem, Population, div);
                        case 7 % SBX
                            valOffspring = Operator_SBX(Population, Problem, proC, disC);
                        case 8 % PM
                            valOffspring = Operator_PM(Population, Problem, proM, disM);
                    end
                    
                    SelectionPool = [Population, valOffspring];
                    [Population, ~, ~] = EnvironmentalSelection(SelectionPool, length(Population), 2); 
                
                else 
                    alpha = 0.3;                   
                    switch operator
                        case 1 % GA 
                                % valOffspring = OperatorGAhalf(Problem, Population(randi(length(Population), 1, Problem.N))); 
                                OffDec  = OperatorGAhalfonly(Problem, Population(randi(length(Population), 1, Problem.N)));
                                No = size(OffDec,1);
                                ParentIdx = randi(length(Population), No, 1);
                                ParentDecRef = reshape([Population(ParentIdx).dec], Problem.D, [])';
                                mask = false(1, Problem.D);
                                Ksel  = max(1, round(alpha * G));
                                selected_groups = randperm(G, Ksel);
                                all_indices = [group_index{selected_groups}];
                                mask(all_indices) = true;
                                OffDec(:, ~mask) = ParentDecRef(:, ~mask);
                                valOffspring = Problem.Evaluation(OffDec);
                            case 2 % DE 
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
                            case 3 % LMOEADS
                                valOffspring = Operator_LMOEADS(Problem, Population, V, Nw, Ns);
                            case 4 % Competitive Learning 
                                valOffspring = CompetitiveLearning(Problem, Population, z, znad, V, Rate);
                            case 5 % Two-stage Sampling
                                valOffspring = TwoStageSampling(Problem, Population, V, Ns1, Ns2);
                            case 6 % MOPSO
                                valOffspring = Operator_MOPSO(Problem, Population, div);
                            case 7 % SBX 
                                OffDec = Operator_SBXonly(Population, Problem, proC, disC);
                                ParentDec   = reshape([Population.dec], Problem.D, [])';
                                mask = false(1, Problem.D);
                                Ksel  = max(1, round(alpha * G));
                                selected_groups = randperm(G, Ksel);
                                all_indices = [group_index{selected_groups}];
                                mask(all_indices) = true;
                                OffDec(:, ~mask) = ParentDec(:, ~mask);
                                valOffspring = Problem.Evaluation(OffDec);
                            case 8 % PM 
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
                    
                    [~,~,Next] = EnvironmentalSelection( [Population,valOffspring],length(Population),2);
                    succ_rate(1,cnt) =  (sum(Next(1:length(Population)))/100) - (sum(Next(length(Population)+1:end))/50);
                    if   succ_rate(1,cnt) >0 
                        currentPopSize = length(Population);
                        rand_number = randperm(currentPopSize);
                        selectCount = min(ceil(currentPopSize/4), currentPopSize);
                        additional_solutions = Population(rand_number(1:selectCount));
                        [Population,~,~] = EnvironmentalSelection( [Population,valOffspring,additional_solutions],length(Population),2);
                    else 
                        [Population,~,~] = EnvironmentalSelection( [Population,valOffspring],length(Population),2);
                    end

                end
                
                average_f1 = log( mean(sum(Population.objs,2)) + eps );
                average_d1 = sum( prctile(Population.objs, 90) - prctile(Population.objs, 10) );
                state_new = [average_f1, average_d1];
                
                delta_f = average_f - average_f1;
                delta_d = average_d1 - average_d;
                hist_f_data(hist_ptr) = delta_f;
                hist_d_data(hist_ptr) = delta_d;
                hist_ptr = hist_ptr + 1;
                hist_len = min(hist_len + 1, max_history);
                if hist_ptr > max_history
                    hist_ptr = 1; 
                end
                if hist_len >= 20
                    valid_f_data = hist_f_data(1:hist_len);
                    valid_d_data = hist_d_data(1:hist_len);
                    f_std = std(valid_f_data);
                    d_std = std(valid_d_data);
                    if f_std < 1e-10, f_std = 1e-10; end
                    if d_std < 1e-10, d_std = 1e-10; end
                    conv_reward = tanh(delta_f / f_std);
                    div_reward = tanh(delta_d / d_std);
                else
                    conv_reward = tanh(delta_f * 1e5);
                    div_reward = tanh(delta_d / 5000);
                end
                w_conv = 0.7;
                w_div = 0.3;
                reward1 = w_conv * conv_reward + w_div * div_reward;

                current_record = [state_old, operator, reward1, state_new];

                Data_len = Data_len + 1;
                if Data_len > capacity
                    Data_buffer(1) = [];
                    Data_buffer{capacity} = current_record;
                    Data_len = capacity;
                else
                    Data_buffer{Data_len} = current_record;
                end
                
                if model_built
                    count = count + 1;
                    if count > 20 
                        % update model 
                        qs=Params.qs;
                        state_dim = length(state_old);
                        temp_data = vertcat(Data_buffer{1:Data_len});
                        use_data = randperm(Data_len, min(20, Data_len));
                        tr_x = temp_data(use_data,1:state_dim+1); 
                        [tr_xx,ps] = mapminmax(tr_x');tr_xx=tr_xx'; 
                        reward1 = testNet(tr_xx,net,Params);
                        reward1=mapminmax('reverse',reward1',qs);reward1=reward1'; 
                        succ = reward1(:,1);
                        tr_yy = temp_data(use_data,state_dim+2)+gama*max(succ);
                        [tr_yy,qs] = mapminmax(tr_yy');tr_yy=tr_yy';
                        Params.ps  = ps;Params.qs=qs;
                        net = updatemodel(tr_xx,tr_yy,Params,net);
                        count = 0;
                    end
                end
                gen = gen+1;
            end
        end
    end
end


% Variable grouping
function group_index = build_groups(D)
    G = 6; 
    gs = floor(D / G);
    group_index = cell(G,1);
    for g = 1:G
        if g < G
            group_index{g} = (g-1)*gs+1 : g*gs;
        else
            group_index{g} = (g-1)*gs+1 : D;
        end
    end
end
