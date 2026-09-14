classdef MMLP
    properties
        n_inputs % 输入层大小 = 决策变量维度(num_D)
        n_outputs % 输出层大小 = 决策变量维度(num_D)
        n_hidden_layers % 隐藏层数量 = 1
        n_hidden_units % 每个隐藏层中的神经元数量 = 20
        hidden_weights % 隐藏层权重
        output_weights % 输出层权重
        learning_rate % 学习率 = 0.5
        momentum % 动量参数 = 0.1
        prev_hidden_weight_delta % 上一次隐藏层权重更新的梯度
        prev_output_weight_delta % 上一次输出层权重更新的梯度
    end
    
    methods
        %% 创建和初始化神经网络的权重矩阵
        function obj = MMLP(n_inputs, n_outputs, n_hidden_layers, n_hidden_units, learning_rate, momentum)
            obj.n_inputs = n_inputs;
            obj.n_outputs = n_outputs;
            obj.n_hidden_layers = n_hidden_layers;
            obj.n_hidden_units = n_hidden_units;
            obj.learning_rate = learning_rate;
            obj.momentum = momentum;
            
            % 初始化隐藏层和输出层权重
            obj.hidden_weights = cell(1, n_hidden_layers); %创建一个1行×n_hidden_layers列的空单元数组，cell 可以灵活存储不同大小的矩阵
            obj.output_weights = randn(n_hidden_units(end), n_outputs); %输出层权重矩阵的行数 = 最后一个隐藏层的神经元数量
            
            % 遍历所有隐藏层，随机初始化每一层的权重矩阵
            for i = 1:n_hidden_layers
                if i == 1
                    input_size = n_inputs; %第1层：输入维度 = n_inputs
                else
                    input_size = n_hidden_units(i-1); %后续层：输入维度 = 前一层的神经元数量
                end
                obj.hidden_weights{i} = randn(input_size, n_hidden_units(i)); %生成该隐藏层的权重矩阵（输入维度*该隐藏层神经元数）
            end
            
            obj.prev_hidden_weight_delta = cell(1, n_hidden_layers); %存储 隐藏层 的历史权重更新
            obj.prev_output_weight_delta = zeros(n_hidden_units(end), n_outputs); %存储 输出层 的历史权重更新

            % 初始化动量参数，用于神经网络训练时的权重更新优化
            for i = 1:n_hidden_layers %为每个隐藏层的动量项分配存储空间，并初始化为零矩阵
                obj.prev_hidden_weight_delta{i} = zeros(size(obj.hidden_weights{i}));
            end
        end
        
        %% 前向传播
        function [output, hidden_outputs] = forward(obj, input)
            hidden_outputs = cell(1, obj.n_hidden_layers); %初始化hidden_outputs，用于存储每个隐藏层的输出
            % 计算每个隐藏层的输出
            for i = 1:obj.n_hidden_layers
                if i == 1
                    hidden_input = input * obj.hidden_weights{i}; %第1隐藏层输入=原始输入 × 第1层权重
                else
                    hidden_input = hidden_outputs{i-1} * obj.hidden_weights{i}; %其他层输入=前一层的输出 × 当前层权重
                end
                hidden_outputs{i} = sigmoid(hidden_input); %应用激活函数计算当前层输出
            end
            % 计算输出层的输出
            output = sigmoid(hidden_outputs{end} * obj.output_weights); %最后一个隐藏层的输出 × 输出层权重
        end

        %% 训练网络（反向传播训练）
        function train(obj, inputs, targets)
            % 向前传播，得到最终输出以及每个隐藏层的输出
            [output, hidden_outputs] = obj.forward(inputs);
            
            % 反向传播，计算输出层和每个隐藏层的误差
            output_error = (output - targets) .* sigmoid_derivative(output); %计算输出层误差
            hidden_errors = cell(1, obj.n_hidden_layers);
            hidden_errors{end} = output_error * obj.output_weights' .* sigmoid_derivative(hidden_outputs{end}); %计算最后一层隐藏层误差，将输出误差反向传播到最后一层隐藏层
            for i = obj.n_hidden_layers-1:-1:1 %逐层向前传播误差：从最后一层隐藏层到第一层隐藏层，依次计算每层的误差
                hidden_errors{i} = hidden_errors{i+1} * obj.hidden_weights{i+1}' .* sigmoid_derivative(hidden_outputs{i});
            end
            
            % 更新输出层权重（动量法）
            obj.prev_output_weight_delta = obj.momentum * obj.prev_output_weight_delta - obj.learning_rate * hidden_outputs{end}' * output_error; %计算权重更新梯度
            obj.output_weights = obj.output_weights + obj.prev_output_weight_delta; %计算新的输出层权重
        
            % 更新隐藏层权重（动量法）
            for i = obj.n_hidden_layers:-1:1
                if i == 1
                    hidden_input = inputs; %第1层用原始输入
                else
                    hidden_input = hidden_outputs{i-1}; %其他层的输入是前一层的输出
                end
                obj.prev_hidden_weight_delta{i} = obj.momentum * obj.prev_hidden_weight_delta{i} - obj.learning_rate * hidden_input' * hidden_errors{i}; %计算权重更新梯度
                obj.hidden_weights{i} = obj.hidden_weights{i} + obj.prev_hidden_weight_delta{i}; %计算新的隐藏层权重
            end
        end

        %% 预测输出
        function output = predict(obj, inputs)
            % 预测给定输入的输出
            [output, ~] = obj.forward(inputs);
        end
        
        % 设置学习率
        function set_learning_rate(obj, learning_rate)
            obj.learning_rate = learning_rate;
        end
        
        % 设置动量参数
        function set_momentum(obj, momentum)
            obj.momentum = momentum;
        end
    end
end

% sigmoid 激活函数
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

% sigmoid 导数
function y = sigmoid_derivative(x)
    y = sigmoid(x) .* (1-sigmoid(x));
end
