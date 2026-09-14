function net = trainNet(x, y, Params, net)
    N = size(x, 1);
    dropP = Params.dropP;
    learnR = Params.learnR;
    W = net.W;
    B = net.B;

    [x1, ~] = dropout_fast(x, dropP(1));
    x2 = x1 * W{1} + B{1};
    x3 = max(0, x2);
    [x4, mask4] = dropout_fast(x3, dropP(2));
    x5 = x4 * W{2} + B{2};
    x6 = tanh(x5);               
    x7 = x6 * W{3} + B{3};

    e = x7 - y;
    dW{3} = x6' * e;
    dB{3} = sum(e, 1);
    dx6 = e * W{3}';

    dx5 = dx6 .* (1 - x6.^2);
    dW{2} = x4' * dx5;
    dB{2} = sum(dx5, 1);
    dx4 = dx5 * W{2}';

    dx3 = zeros(size(x3), 'like', x4);
    if ~isempty(mask4)
        dx3(mask4) = dx4(mask4) / (1 - dropP(2));
    end

    dx2 = dx3;
    dx2(x3 <= 0) = 0;

    dW{1} = x1' * dx2;
    dB{1} = sum(dx2, 1);

    decay = 1e-5;
    for k = 1:3
        W{k} = W{k} - (decay * W{k} + dW{k}) / N * learnR;
        B{k} = B{k} - dB{k} / N * learnR;
    end
    net.W = W;
    net.B = B;
end


function [x, mask] = dropout_fast(x, dropP)
    if dropP == 0
        mask = [];
        return;
    end
    mask = rand(size(x), 'like', x) > dropP;
    x(~mask) = 0;
    x = x / (1 - dropP);
end