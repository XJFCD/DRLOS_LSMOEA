% function [net, Params] = trainmodel(tr_x, tr_y, Params)
% 
%     V=size(tr_x,2);M=size(tr_y,2);
%     neuronN=40;N=size(tr_x,1);bias1=0.1;bias2=0;
%     Params.neuronN=neuronN;
%     Params.dropP=[0.2,0.5];
%     Params.decay=1e-05;Params.learnR=0.01;
%     batchsize=V;Params.batchsize=batchsize;
%     run=80000;Params.round=run;
% 
%     %W
%     flag=0;
%     W{1}=iniA(V,neuronN,flag,bias2);W{2}=iniA(neuronN,neuronN,flag,bias2);
%     W{3}=iniA(neuronN,M,flag,bias2);
%     %B
%     flag=1;
%     B{1}=iniA(1,neuronN,flag,bias1);B{2}=iniA(1,neuronN,flag,bias2);
%     array=iniA(1,M,flag,bias2);B{3}=array(1:M);
%     %net
%     net.W=W;net.B=B;
%     index=round(1+rand(1,run*batchsize)*(N-1));
%     for j=1:run
%         x=tr_x(index((j-1)*batchsize+1:j*batchsize),:);
%         y=tr_y(index((j-1)*batchsize+1:j*batchsize),:);      
%         net=trainNet(x, y, Params, net);
%     end
% end
function [net, Params] = trainmodel(tr_x, tr_y, Params)
    % 获取维度
    [N, V] = size(tr_x);
    M = size(tr_y, 2);

    % 超参数（保持不变）
    neuronN = 40;
    bias1 = 0.1;
    bias2 = 0;
    Params.neuronN = neuronN;
    Params.dropP = [0.2, 0.5];
    Params.decay = 1e-5;
    Params.learnR = 0.01;
    run = 80000;             % 总迭代次数
    batchsize = V;           % 你坚持保持不变

    % ========= 1. 数据自动移到 GPU =========
    if canUseGPU
        tr_x = gpuArray(tr_x);
        tr_y = gpuArray(tr_y);
        fprintf('trainmodel: 使用 GPU 加速\n');
    end

    % ========= 2. 初始化权重（与设备一致）=========
    flag = 0;
    W{1} = iniA(V, neuronN, flag, bias2);
    W{2} = iniA(neuronN, neuronN, flag, bias2);
    W{3} = iniA(neuronN, M, flag, bias2);
    flag = 1;
    B{1} = iniA(1, neuronN, flag, bias1);
    B{2} = iniA(1, neuronN, flag, bias2);
    array = iniA(1, M, flag, bias2);
    B{3} = array(1:M);

    % 转到 GPU（如果可用）
    if isa(tr_x, 'gpuArray')
        for k = 1:3
            W{k} = gpuArray(W{k});
            B{k} = gpuArray(B{k});
        end
    end

    net.W = W;
    net.B = B;

    % ========= 3. 预生成随机索引 =========
    total_iters = run * batchsize;
    idx_all = randi(N, total_iters, 1);

    % ========= 4. 预分配 batch 内存（复用）=========
    x_batch = zeros(batchsize, V, 'like', tr_x);
    y_batch = zeros(batchsize, M, 'like', tr_y);

    % ========= 5. 主训练循环 =========
    for j = 1:run
        start_idx = (j-1)*batchsize + 1;
        end_idx   = j*batchsize;
        idx = idx_all(start_idx:end_idx);

        % 复用内存，避免分配新数组
        x_batch(:,:) = tr_x(idx, :);
        y_batch(:,:) = tr_y(idx, :);

        % 调用优化后的 trainNet（见下方）
        net = trainNet_fast(x_batch, y_batch, Params, net);
    end
end