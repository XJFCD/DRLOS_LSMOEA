function net = trainNet(x, y, Params, net)
    N = size(x, 1);
    dropP = Params.dropP;
    learnR = Params.learnR;
    W = net.W;
    B = net.B;

    % ========= 前向传播（无 repmat，使用广播）=========
    [x1, ~] = dropout_fast(x, dropP(1));
    x2 = x1 * W{1} + B{1};
    x3 = max(0, x2);
    [x4, mask4] = dropout_fast(x3, dropP(2));
    x5 = x4 * W{2} + B{2};
    x6 = tanh(x5);               % 替代手动指数
    x7 = x6 * W{3} + B{3};

    % ========= 反向传播（全向量化）=========
    e = x7 - y;
    dW{3} = x6' * e;
    dB{3} = sum(e, 1);
    dx6 = e * W{3}';

    dx5 = dx6 .* (1 - x6.^2);
    dW{2} = x4' * dx5;
    dB{2} = sum(dx5, 1);
    dx4 = dx5 * W{2}';

    % ★★★ 关键优化：直接用 mask 进行向量化反向传播 ★★★
    dx3 = zeros(size(x3), 'like', x4);
    if ~isempty(mask4)
        dx3(mask4) = dx4(mask4) / (1 - dropP(2));
    end

    dx2 = dx3;
    dx2(x3 <= 0) = 0;

    dW{1} = x1' * dx2;
    dB{1} = sum(dx2, 1);

    % 权重衰减 + 更新
    decay = 1e-5;
    for k = 1:3
        W{k} = W{k} - (decay * W{k} + dW{k}) / N * learnR;
        B{k} = B{k} - dB{k} / N * learnR;
    end
    net.W = W;
    net.B = B;
end

% ========= 优化后的 dropout（GPU/CPU 兼容，返回掩码而非索引）=========
function [x, mask] = dropout_fast(x, dropP)
    if dropP == 0
        mask = [];
        return;
    end
    % 使用 'like', x 确保随机数组类型/设备与 x 一致
    mask = rand(size(x), 'like', x) > dropP;
    x(~mask) = 0;
    x = x / (1 - dropP);
end
% function net = trainNet(x, y, Params, net)
%     % 将CPU版本修改为GPU版本
%     % 将输入数据移到 GPU（只需做一次，在调用这个函数之前）
%     % 或者在函数内部转换：
%     if ~isa(x, 'gpuArray')
%         x = gpuArray(x);
%         y = gpuArray(y);
%         % 同时需要把 net.W 和 net.B 也移到 GPU
%         for k = 1:3
%             net.W{k} = gpuArray(net.W{k});
%             net.B{k} = gpuArray(net.B{k});
%         end
%     end
% 
%     N=size(x,1);
%     dropP=Params.dropP;learnR=Params.learnR;
%     W=net.W;B=net.B;
% 
%     %% forward
%     % [x1,~]=dropout(x,dropP(1));%N*V
%     [x1, ~] = dropout_gpu(x, dropP(1));  % 改写 dropout 函数
%     x2=x1*W{1}+repmat(B{1},N,1);%N*neuronN
%     x3=max(0,x2);%ReLU
%     % [x4, index1, index2]=dropout(x3,dropP(2));
%     [x4, index1, index2] = dropout_gpu(x3, dropP(2));
%     x5=x4*W{2}+repmat(B{2},N,1);
%     % x6=(exp(x5)-exp(-x5))./(exp(x5)+exp(-x5));%Sigmoid%N*neuronN
%     x6 = tanh(x5);  % 直接用 tanh，比 exp 版本更快更稳定
%     x7=x6*W{3}+repmat(B{3},N,1);%N*M
% 
%     %% backward
%     %cost_loss=sum(sum(0.5*(x7-y).^2));
%     %fc x6-x7
%     e=x7-y;%N*M
%     dW{3}=x6'*e;%neuronN*M
%     % dB{3}=sum(e);%1*M
%     dB{3} = sum(e, 1);  % 注意用 sum(...,1) 保持维度
%     dx6=e*W{3}';%N*neuronN
%     %sigmoid x5-x6
%     dx5=dx6.*(1-x6.^2);%N*neuronN
%     %fc x4-x5
%     dW{2}=x4'*dx5;%neuronN*neuronN
%     % dB{2}=sum(dx5);%1*neuronN
%     dB{2} = sum(dx5, 1);
%     dx4=dx5*W{2}';%N*neuronN
%     %dp x3-x4
%     %tic;
%     % dx3=zeros(size(x3));%N*neuronN
%     % for i=1:length(index1)
%     %     dx3(index1(i),index2(i))=dx4(index1(i), index2(i))/(1-dropP(2));
%     % end
%     dx3 = zeros(size(x3), 'like', x4);
%     linear_idx = sub2ind(size(x3), index1, index2);
%     dx3(linear_idx) = dx4(linear_idx) / (1 - dropP(2));
%     %ReLU x2-x3
%     dx2=dx3;%N*neuronN
%     dx2(x3<=0)=0;%N*neuronN
%     %fc x1-x2
%     dW{1}=x1'*dx2;%V*neuronN
%     % dB{1}=sum(dx2);%1*neuronN
%     dB{1} = sum(dx2, 1);
% 
%     decay=1e-05;
%     for k=1:3
%         W{k}=W{k}-(decay*W{k}+dW{k})/N*learnR;
%         B{k}=B{k}-dB{k}/N*learnR;
%     end
%     net.W=W;net.B=B;
% end
% 
% % 辅助函数：GPU 兼容的 dropout
% function [out, idx1, idx2] = dropout_gpu(x, dropP)
%     if dropP == 0
%         out = x;
%         idx1 = []; idx2 = [];
%         return;
%     end
%     mask = rand(size(x), 'like', x) > dropP;  % 'like', x 保持 GPU 类型
%     out = x .* mask / (1 - dropP);
%     [idx1, idx2] = find(mask);  % find 在 GPU 上也支持
% end