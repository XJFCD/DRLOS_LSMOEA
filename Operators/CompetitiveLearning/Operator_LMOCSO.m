function [OffDec,OffVel] = Operator_LMOCSO(Problem,Loser,Winner,Rate)
% The competitive swarm optimizer of LMOCSO 基于竞争的多目标粒子群优化算子
% 输入：Problem：优化问题定义
%           Loser：竞争中的失败者（需改进的解）
%           Winner：竞争中的胜利者（优质解）
%           Rate：控制探索与开发的参数
% 输出：OffDec：新生成的决策变量
%           OffVel：新生成的速度向量（用于粒子群算法）

%  Copyright (C) 2021 Xu Yang
%  Xu Yang <xuyang.busyxu@qq.com> or <xuyang369369@gmail.com>

    %% Parameter setting
    LoserDec  = Loser.decs; % 失败者的决策变量
    WinnerDec = Winner.decs; % 胜利者的决策变量
    [N,D]     = size(LoserDec); % N:解数量, D:变量维度
	LoserVel  = Loser.adds(zeros(N,D)); % 失败者的速度（若无则初始化为0）
    WinnerVel = Winner.adds(zeros(N,D)); % 胜利者的速度
    
    %% Competitive swarm optimizer 竞争粒子群更新
    % 失败者向胜利者方向移动，同时保留部分原速度（惯性）
    r1     = repmat(rand(N,1),1,D); % 随机系数1（N*D）
    r2     = repmat(rand(N,1),1,D); % 随机系数2（N*D） 
    OffVel = r1.*LoserVel + r2.*(WinnerDec-LoserDec); % 速度更新
    OffDec = LoserDec + OffVel; % 位置更新
    
    % 增强探索（早期阶段）
    if Problem.FE/Problem.maxFE < Rate %如果在进化早期（当前评估次数与最大评估次数的比值小于Rate）
        LoserVel1 = rand(N,D); %增加随机扰动
        OffVel1 = r1.*LoserVel1 + r2.*(WinnerDec-LoserDec); %计算额外的子代速度
        OffDec1 = LoserDec + OffVel1 + r1.*(OffVel1-LoserVel1); %计算额外的子代位置
        % 将这部分子代加入原有子代中
        OffDec = [OffDec;OffDec1];
        OffVel = [OffVel;OffVel1];
    end
    
    %% Add the winners 保留优胜者
    % 将Winner的决策变量和速度直接加入子代群体中
    OffDec = [OffDec;WinnerDec];
    OffVel = [OffVel;WinnerVel];
 
    %% Polynomial mutation 多项式变异
    % 对子代进行多项式变异以增加多样性
    [N,D] = size(OffDec);
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    disM  = 20; %使用多项式分布进行变异，变异指数为20，越大，变异越集中于父代附近
    Site  = rand(N,D) < 1/D; %变异概率为1/D
    mu    = rand(N,D);
    temp  = Site & mu<=0.5; %当随机数mu ≤ 0.5时，产生一个偏向较小值的变异量，使得解向左移动
    OffDec       = max(min(OffDec,Upper),Lower); %边界处理
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp  = Site & mu>0.5; %当随机数mu ≤ 0.5时，产生一个偏向较大值的变异量，使得解向右移动
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));          
end