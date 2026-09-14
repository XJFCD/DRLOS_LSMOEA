function [P1,P2,A,B] = classification1(Population,z,znad,V)
% z：理想点，目标空间中所有目标函数的最小值
% znad：反理想点/最差点，目标空间中所有目标函数的最差值
% V：每个子问题方向的权重向量
    [PopObj,z,znad] = Normalization(Population.objs,z,znad); %目标值归一化&参考点更新 
    tFrontNo = tNDSort(PopObj,V); %对种群进行非支配排序
    tNum = size(tFrontNo,2); %获取种群中个体的数量 N
    [N,M] = size(Population.objs);
    front = unique(tFrontNo)'; % 获取所有前沿编号
    remain =N/2; % 需要选入 P1的解数量（总解数的一半）
    i = 1;
    q = sum(tFrontNo==front(i)); %统计非支配解的数量
    s = [remain>0,remain>=q]; 
    Next = zeros(1,N); %存储选中的解的位置
    Next1 = zeros(1,N); %存储未被选中的解的位置
    % 选择优质解 (P1)
    % 按前沿顺序（从最优到最差）选择解，直到满足 N/2 的数量要求。
    while sum(s)==2
        Next0 = find(tFrontNo==front(i));
        Next(Next0) = true; %将找到的非支配解标记为true
        remain =remain -sum(tFrontNo==front(i)); %计算还需要选多少个解
        i = i+1;
        q = sum(tFrontNo==front(i)); %统计下一个前沿解的数量
        s = [remain >0,remain >=q];
    end
    % 处理剩余解
    if(remain >0)
        Last = find(tFrontNo==front(i)); %获取要加入的最后一层前沿
        fitness = calFitness(Population(find(tFrontNo==front(i))).objs);
        [~,Rank] = sort(fitness,'descend');
        Next(Last(Rank(1:remain))) = true; %对最后一个未完全选中的前沿，根据适应度进一步筛选
        Next1(Last(Rank(remain+1:q))) = true;
    end
    % 构建子群
     Next1(find(Next==0))=true; %获取未被选中的解
     P1 = Population(logical(Next')); %P1：保留的优质解（前半部分）
     P2 =Population(logical(Next1')); %P2：次优解（后半部分）
     A = find(Next==1); % 获取被选入P1的个体索引
     B = find(Next1==1); % 获取被选入P2的个体索引
end

% Do theta-non-dominated sorting θ-非支配排序，用于基于分解的多目标进化算法
function tFrontNo = tNDSort(PopObj,W)
    % Do theta-non-dominated sorting θ-非支配排序，用于基于分解的多目标进化算法
    % 核心思想：θ-支配关系通过两个距离指标(d₁和d₂)和惩罚参数θ来平衡；
    % 收敛性(d₁)：解到参考向量的投影距离；
    % 多样性(d₂)：解与参考向量的垂直距离；
    % θ惩罚：控制多样性对排序的影响强度。
    % 输入：PopObj: 种群目标值矩阵；W：每个解方向的权重向量
    % 输出：tFrontNo: 每个个体的θ-非支配前沿编号 (1×N向量)

    N  = size(PopObj,1); %种群个数
    NW = size(W,1); %权重向量矩阵的行数

    %% Calculate the d1 and d2 values for each solution to each weight 计算每个权重方向的每个解的d1和d2值
    normP  = sqrt(sum(PopObj.^2,2)); % 计算每个解到原点的欧氏距离
    Cosine = 1 - pdist2(PopObj,W,'cosine'); % 计算解与每个权重向量的夹角余弦值
    d1     = repmat(normP,1,size(W,1)).*Cosine; % 到每个权重向量方向上的投影距离(收敛性指标)
    d2     = repmat(normP,1,size(W,1)).*sqrt(1-Cosine.^2); %到每个权重向量方向的垂直距离(多样性指标)
    % d2(i,j) 表示：第i个解 与 第j个参考向量 的垂直距离；d2 是一个 N×NW 的矩阵（N=解数量，NW=参考向量数量）
    
    %% Clustering 聚类
    [~,class] = min(d2,[],2); % 对d2按行取最小，只返回每个解所属的参考向量索引，实现将每个解分配到最近的参考向量(垂直距离最小)
    
    %% Sort 分类
    theta = zeros(1,NW) + 5; % 默认θ=5，普通方向：θ=5 (平衡收敛性和多样性)
    theta(sum(W>1e-4,2)==1) = 1e6; % 当 W 的某一行仅有一个元素显著大于零时，将θ设为极大值(1e6)，即标记边界方向 (用于惩罚超出边界的)
    tFrontNo = zeros(1,N); %初始化每个解的前沿编号为0
    for i = 1 : NW %遍历所有权重向量
        C = find(class==i); % 找到分配到第 i 个参考向量的解
        [~,rank] = sort(d1(C,i)+theta(i)*d2(C,i)); % 对计算出的距离排序：d1(C,i) 解C到参考向量i的投影距离 + 该权重方向的θ * d2(C,i) 解C到参考向量i的垂直距离
        tFrontNo(C(rank)) = 1 : length(C); % 按照排序索引对C中解排序，然后为tFrontNo中的解分配前沿编号
    end
end

function Fitness = calFitness(PopObj)
% Calculate the fitness by shift-based density
    N      = size(PopObj,1);
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    Dis    = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end
