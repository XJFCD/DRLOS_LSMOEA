function Offspring = Operator_PMonly(Population, Problem, proM, disM)
% Operator_PM - 多项式变异算子
% 输入：
%   Problem: 优化问题对象
%   Population: 父代种群（SOLUTION对象）
%   proM: 每个个体期望发生变异的变量个数（默认1）
%   disM: 多项式变异的分布指数（默认20）
% 输出：
%   Offspring: 子代种群（SOLUTION对象）
    
    % 参数处理
    if nargin < 3 || isempty(proM)
        proM = 1;
    end
    if nargin < 4 || isempty(disM)
        disM = 20;
    end
    % 提取决策变量
    if isa(Population(1), 'SOLUTION')
        evaluated = true;
        Parent = Population.decs;
    else
        evaluated = false;
        Parent = Population;
    end

    [N,D]   = size(Parent);
    OffDec = Parent; % 初始化为父代
    
    %% Polynomial mutation多项式变异
    Lower = repmat(Problem.lower, N,1);
    Upper = repmat(Problem.upper, N,1);
    Site  = rand(N,D) < proM/D; % 确定变异位置
    mu    = rand(N,D);
    % 多项式变异计算
    temp  = Site & mu<=0.5;
    OffDec       = min(max(OffDec,Lower),Upper);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

    % 边界处理
    OffDec = max(min(OffDec, Upper), Lower);
    
    % 返回子代
    Offspring = OffDec;
    % if evaluated
    %     Offspring = Problem.Evaluation(OffDec);
    % else
    %     Offspring = OffDec;
    % end
end