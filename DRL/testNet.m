function x7 = testNet(x, net, ~)
    
    W = net.W;
    B = net.B;
    
    if isa(W{1}, 'gpuArray') && ~isa(x, 'gpuArray')
        x = gpuArray(x);
    end
    x1 = x;                                    
    x2 = x1 * W{1} + B{1};                
    x3 = max(0, x2);
    x4 = x3;                                  
    x5 = x4 * W{2} + B{2};
    x6 = tanh(x5);                           
    x7 = x6 * W{3} + B{3};
    
    if isa(x7, 'gpuArray')
        x7 = gather(x7);
    end
end