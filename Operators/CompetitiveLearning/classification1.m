function [P1,P2,A,B] = classification1(Population,z,znad,V)

    [PopObj,z,znad] = Normalization(Population.objs,z,znad);  
    tFrontNo = tNDSort(PopObj,V); 
    tNum = size(tFrontNo,2); 
    [N,M] = size(Population.objs);
    front = unique(tFrontNo)'; 
    remain =N/2; 
    i = 1;
    q = sum(tFrontNo==front(i)); 
    s = [remain>0,remain>=q]; 
    Next = zeros(1,N);
    Next1 = zeros(1,N); 

    while sum(s)==2
        Next0 = find(tFrontNo==front(i));
        Next(Next0) = true; 
        remain =remain -sum(tFrontNo==front(i)); 
        i = i+1;
        q = sum(tFrontNo==front(i)); 
        s = [remain >0,remain >=q];
    end

    if(remain >0)
        Last = find(tFrontNo==front(i)); 
        fitness = calFitness(Population(find(tFrontNo==front(i))).objs);
        [~,Rank] = sort(fitness,'descend');
        Next(Last(Rank(1:remain))) = true; 
        Next1(Last(Rank(remain+1:q))) = true;
    end

     Next1(find(Next==0))=true;
     P1 = Population(logical(Next')); 
     P2 =Population(logical(Next1')); 
     A = find(Next==1); 
     B = find(Next1==1); 
end


function tFrontNo = tNDSort(PopObj,W)

    N  = size(PopObj,1); 
    NW = size(W,1); 

    %% Calculate the d1 and d2 values for each solution to each weight 
    normP  = sqrt(sum(PopObj.^2,2));
    Cosine = 1 - pdist2(PopObj,W,'cosine'); 
    d1     = repmat(normP,1,size(W,1)).*Cosine; 
    d2     = repmat(normP,1,size(W,1)).*sqrt(1-Cosine.^2); 
    
    %% Clustering 
    [~,class] = min(d2,[],2); 
    
    %% Sort 
    theta = zeros(1,NW) + 5; 
    theta(sum(W>1e-4,2)==1) = 1e6; 
    tFrontNo = zeros(1,N); 
    for i = 1 : NW
        C = find(class==i); 
        [~,rank] = sort(d1(C,i)+theta(i)*d2(C,i)); 
        tFrontNo(C(rank)) = 1 : length(C);
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
