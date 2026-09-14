function Arc = DirectedSampling2(GDV,Ns,W,Problem) % ,lambda增加lambda参数
    %fprintf('DirectedSampling2 - Lambda received: %.4f\n', lambda);

    % 水平采样方法
    % 输入：GDV：非支配解集合，Ns：采样数量30，W：参考权重向量（用于分解目标空间），Problem：优化问题定义
    % 输出：Arc：新生成的候选解集合（经过非支配筛选）
    [~,m] = size(W); % 获取目标变量维度
    Z = zeros(1,m);  % 理想点（初始化为0）
    PopX = GDV.decs; % 提取非支配解的决策变量
    a = [];b = [];
    [Nr,D] = size(PopX); % Nr:解数量, D:决策变量维度
    Nc = ceil(Nr); %将Nr向上取整
    
    % 从非支配解中随机选取 Nc 对解（a和b)，每对解将定义一个搜索方向
    for i = 1:Nc
        a = [a;PopX(randi([1,Nr],1),:)]; 
        b = [b;PopX(randi([1,Nr],1),:)]; 
    end

    % 计算搜索方向与采样步长
    vmax = sqrt(sum((Problem.upper-Problem.lower).^2,2)); % 最大可能步长
    vmin = 0; % 最小步长
    centerPoint = (a+b)/2; % 计算每对的中心点
    center = a-b; % 方向向量
    Directnorm = sqrt(sum(center.^2,2)); % 方向向量的模
    Direct = center./repmat(Directnorm,1,D); % 单位方向向量
    v0 = vmin + (rand(Ns,Nc)*2-1)*vmax*0.7; % 随机步长，在 [-0.7*vmax, 0.7*vmax] 范围内，0.7为衰减系数，避免搜索过于激进
    %v0 = vmin + (rand(Ns,Nc)*2-1)*vmax*lambda; %新增：lambda参数
    PopNew = GeneratingSampledSolution1(centerPoint,v0,Direct,Z,Problem);% 生成新解
    Arc = PopNew((NDSort(PopNew.objs,1)==1));  % 保留Rank 1的解   
end

function  PopNew = GeneratingSampledSolution1(centerPoint,v0,Direct,Z,Problem)
    [Ns,VD] = size(v0); %步长v0的维度
    D = size(Problem.upper,2); %问题的变量维度
    m = size(Z,2); 
    upper = Problem.upper;lower = Problem.lower;
    PopNew = [];
    for i = 1:Ns
        PopX = centerPoint + repmat(v0(i,1:VD)',1,D).* Direct(1:VD,:); % 从中心点沿方向向量和随机步长生成新解
        PopX = max(min(repmat(upper,size(PopX,1),1),PopX),repmat(lower,size(PopX,1),1)); % 边界处理
        Pop =  Problem.Evaluation(PopX); % 评估解  
        PopNew = [PopNew,Pop]; %将新生成的候选解 Pop 追加到集合 PopNew 中
    end
end
