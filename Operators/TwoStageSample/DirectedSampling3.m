function Arc = DirectedSampling3(a,b,Ns,W,Problem)
    %fprintf('DirectedSampling2 - Lambda received: %.4f\n', lambda);

    % 垂直采样方法
    % 输入：a: 预测得到的新非支配解；b: 原始非支配解，Ns：采样数量30，W：参考权重向量（用于分解目标空间），Problem：优化问题定义
    % 输出：Arc：新生成的候选解集合（经过非支配筛选）
    [~,m] = size(W);
    Z = zeros(1,m);
    a = a.decs;
    b= b.decs;
    [Nr,D] = size(a);
    Nc = ceil(Nr);
    
    % 计算搜索方向与采样步长
    vmax = sqrt(sum((Problem.upper-Problem.lower).^2,2));
    vmin = 0;
    center = a-b;
    Directnorm = sqrt(sum(center.^2,2));
    Direct = center./repmat(Directnorm,1,D);
    v0 = vmin + (rand(Ns,Nc)*2-1)*vmax;

    PopNew = GeneratingSampledSolution1(a,v0,Direct,Z,Problem);
    Arc = PopNew((NDSort(PopNew.objs,1)==1));
end

function  PopNew = GeneratingSampledSolution1(centerPoint,v0,Direct,Z,Problem)
    [Ns,VD] = size(v0);
    D = size(Problem.upper,2);
    m = size(Z,2);
    upper = Problem.upper;lower = Problem.lower;
    PopNew = [];
    for i = 1:Ns
        PopX = centerPoint + repmat(v0(i,1:VD)',1,D).* Direct(1:VD,:); % 从a点沿方向向量和随机步长生成新解
        PopX = max(min(repmat(upper,size(PopX,1),1),PopX),repmat(lower,size(PopX,1),1));
        Pop =  Problem.Evaluation(PopX);
        PopNew = [PopNew,Pop];
    end
end
