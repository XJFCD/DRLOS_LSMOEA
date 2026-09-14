function [PopObj,z,znad] = Normalization(PopObj,z,znad)
% Normalize the population and update the ideal point and the nadir point
% 目标值归一化 和 参考点更新 
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(PopObj);

    %% Update the ideal point 更新理想点
    z = min(z,min(PopObj,[],1)); %取历史理想点 z 和当前种群最小值中的更小值
    
    %% Update the nadir point 更新反理想点
    % Identify the extreme points 识别极端点
    W = zeros(M) + 1e-6;
    W(logical(eye(M))) = 1; % 创建单位对角矩阵
    ASF = zeros(N,M);
    for i = 1 : M
        ASF(:,i) = max(abs((PopObj-repmat(z,N,1))./(repmat(znad-z,N,1)))./repmat(W(i,:),N,1),[],2); %计算每个解在各个权重向量方向上的表现
    end
    [~,extreme] = min(ASF,[],1); %极端点：每个目标方向上ASF值最小的解
    % Calculate the intercepts 计算超平面截距
    Hyperplane = (PopObj(extreme,:)-repmat(z,M,1))\ones(M,1);
    a = (1./Hyperplane)' + z; %通过极端点拟合超平面，求各目标轴截距作为反理想点
    if any(isnan(a)) || any(a<=z)
        a = max(PopObj,[],1); % 当超平面计算失败时，直接取种群最大值
    end
    znad = a;
    
    %% Normalize the population 目标值归一化
    PopObj = (PopObj-repmat(z,N,1))./(repmat(znad-z,N,1)); 
end