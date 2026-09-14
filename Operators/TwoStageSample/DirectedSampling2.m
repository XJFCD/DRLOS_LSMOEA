function Arc = DirectedSampling2(GDV,Ns,W,Problem)

    [~,m] = size(W);
    Z = zeros(1,m);  
    PopX = GDV.decs; 
    a = [];b = [];
    [Nr,D] = size(PopX); 
    Nc = ceil(Nr); 
    
    for i = 1:Nc
        a = [a;PopX(randi([1,Nr],1),:)]; 
        b = [b;PopX(randi([1,Nr],1),:)]; 
    end

    vmax = sqrt(sum((Problem.upper-Problem.lower).^2,2)); 
    vmin = 0; 
    centerPoint = (a+b)/2; 
    center = a-b; 
    Directnorm = sqrt(sum(center.^2,2)); 
    Direct = center./repmat(Directnorm,1,D); 
    v0 = vmin + (rand(Ns,Nc)*2-1)*vmax*0.7; 
    %v0 = vmin + (rand(Ns,Nc)*2-1)*vmax*lambda; 
    PopNew = GeneratingSampledSolution1(centerPoint,v0,Direct,Z,Problem);
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
