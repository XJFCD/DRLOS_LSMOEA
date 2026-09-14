function [net, Params] = trainmodel(tr_x, tr_y, Params)

    [N, V] = size(tr_x);
    M = size(tr_y, 2);
    neuronN = 40;
    bias1 = 0.1;
    bias2 = 0;
    Params.neuronN = neuronN;
    Params.dropP = [0.2, 0.5];
    Params.decay = 1e-5;
    Params.learnR = 0.01;
    run = 80000;             
    batchsize = V;          

    if canUseGPU
        tr_x = gpuArray(tr_x);
        tr_y = gpuArray(tr_y);
    end

    flag = 0;
    W{1} = iniA(V, neuronN, flag, bias2);
    W{2} = iniA(neuronN, neuronN, flag, bias2);
    W{3} = iniA(neuronN, M, flag, bias2);
    flag = 1;
    B{1} = iniA(1, neuronN, flag, bias1);
    B{2} = iniA(1, neuronN, flag, bias2);
    array = iniA(1, M, flag, bias2);
    B{3} = array(1:M);

    if isa(tr_x, 'gpuArray')
        for k = 1:3
            W{k} = gpuArray(W{k});
            B{k} = gpuArray(B{k});
        end
    end

    net.W = W;
    net.B = B;

    total_iters = run * batchsize;
    idx_all = randi(N, total_iters, 1);

    x_batch = zeros(batchsize, V, 'like', tr_x);
    y_batch = zeros(batchsize, M, 'like', tr_y);

    for j = 1:run
        start_idx = (j-1)*batchsize + 1;
        end_idx   = j*batchsize;
        idx = idx_all(start_idx:end_idx);
        x_batch(:,:) = tr_x(idx, :);
        y_batch(:,:) = tr_y(idx, :);
        net = trainNet(x_batch, y_batch, Params, net);
    end
end
