% function net = updatemodel(tr_x, tr_y, Params, net)
% 
%     [N,V]=size(tr_x);
%     batchsize=V;
%     run=8000;
%     index=round(1+rand(1,run*batchsize)*(N-1));
%     for j=1:run
%         x=tr_x(index((j-1)*batchsize+1:j*batchsize),:);
%         y=tr_y(index((j-1)*batchsize+1:j*batchsize),:);      
%         net=trainNet(x, y, Params, net);
%     end
% end

function net = updatemodel(tr_x, tr_y, Params, net)
    [N, V] = size(tr_x);
    M = size(tr_y, 2);
    run = 8000;          
    batchsize = V;       

    if isa(net.W{1}, 'gpuArray') && ~isa(tr_x, 'gpuArray')
        tr_x = gpuArray(tr_x);
        tr_y = gpuArray(tr_y);
        fprintf('updatemodel: GPU\n');
    end

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