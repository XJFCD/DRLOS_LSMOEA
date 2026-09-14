% function x7 = testNet(x, net, Params)
% 
%     N=size(x,1);
%     W=net.W;B=net.B;
%     dropP=Params.dropP;
%     %% forward
%     [x1,~]=dropout(x,dropP(1));%N*V
%     x2=x1*W{1}+repmat(B{1},N,1);%N*neuronN
%     x3=max(0,x2);%ReLU
%     [x4, ~]=dropout(x3,dropP(2));
%     x5=x4*W{2}+repmat(B{2},N,1);
%     x6=(exp(x5)-exp(-x5))./(exp(x5)+exp(-x5));%Sigmoid%N*neuronN
%     x7=x6*W{3}+repmat(B{3},N,1);%N*M
% end

function x7 = testNet(x, net, ~)
    % 重要：测试时不应该使用 dropout！
    % 直接使用训练好的网络进行推理
    
    W = net.W;
    B = net.B;
    
    % 如果输入是 CPU 数据但网络在 GPU 上，需要转换
    if isa(W{1}, 'gpuArray') && ~isa(x, 'gpuArray')
        x = gpuArray(x);
    end
    
    % 前向传播（无 dropout，无 repmat）
    x1 = x;                                      % 无 dropout
    x2 = x1 * W{1} + B{1};                       % 广播替代 repmat
    x3 = max(0, x2);
    x4 = x3;                                     % 无 dropout
    x5 = x4 * W{2} + B{2};
    x6 = tanh(x5);                               % 内置 tanh
    x7 = x6 * W{3} + B{3};
    
    % 如果结果是 GPU 数组，转回 CPU（可选）
    if isa(x7, 'gpuArray')
        x7 = gather(x7);
    end
end