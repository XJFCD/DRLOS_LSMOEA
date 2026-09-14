function Arc = DirectedSampling3(a,b,Ns,W,Problem)

    [~,m] = size(W);
    Z = zeros(1,m);
    a = a.decs;
    b= b.decs;
    [Nr,D] = size(a);
    Nc = ceil(Nr);
    
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
        PopX = centerPoint + repmat(v0(i,1:VD)',1,D).* Direct(1:VD,:);
        PopX = max(min(repmat(upper,size(PopX,1),1),PopX),repmat(lower,size(PopX,1),1));
        Pop =  Problem.Evaluation(PopX);
        PopNew = [PopNew,Pop];
    end
end
