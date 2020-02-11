function [label,class,K,Center]=KK2(data,AA,Max)
%% %%函数作用如同KK1%%
%% %%只是迭代的代价函数改变了%%%%
[n,c]=size(data); %求输入数据点的个数m
%data1=data;
X=data(:,1:c-1);
maxK=floor(sqrt(n));%确定迭代的最大聚类块数为10%数据总个数
cost=zeros(maxK-1,1);
cost1=zeros(maxK-1,1);%代价
cost2=zeros(maxK-1,1);
%% 求数据集的直径
%  Dists = pdist(X,'euclidean');
%  Max=max(max(Dists));
Center2=cell(maxK-1,1);%中心点
t=1;
%% 找到聚类误差代价最小的K值
for K=2:floor(sqrt(n))
    maxdiameter=cell(K,1);
    %Q=zeros(K,c-1);
    class_distence=cell(K,1);
    class=cell(K,1);
    A=cell(K,1);
    %% 等间隔选取初始聚类中心点
% for i=1:K
%    % C(i,:)=X(randi(m,1),:);     %随机初始化聚类中心%randi(n,1)产生一个1到n的伪随机整数
%    %b=randi(n,1);
%    b=floor((n.*i)./(K+1));% 选取中心点的标号（例如100点聚成2类，就选取了33和66两个点作为初始中心点）
%    Q(i,:)=X(b,:);
% end
% matrix=Q;%初始中心矩阵
% [idx,C]= kmeans(X,K,'start',matrix);%调用了自带的Kmeans算法，运行较快一些
%Center{t,1}=C;
[idx,C]= kmeans(X,K);
Center2{t,1}=C;
cluster_cost=0;%每一块的代价初始值赋为0
purity_cost=0;
%% 计算聚类块数为K时的总代价
for i=1:K
    class{i,1}=data(idx==i,:);%属于第i块的数据
    A{i,1}=AA(idx==i,:);
    A{i,1}=A{i,1}(:,end);
    b=1:max(A{i,1});
    Max_num=max(histc(A{i,1},b));%%找到块中已经预测，购买和未购买出现相同标签的最大数量值
    [p,~]=size(class{i,1});
    %% 计算第i块的直径，并求出相应的误差
    class_distence{i,1}=X(idx==i,:);
     Dists = pdist(class_distence{i,1},'euclidean');
     maxdiameter{i,1}=max(max(Dists));
     [phi]=badblock(maxdiameter,Max,K);
%       normdis=maxdiameter/Max;
%      phi=(-0.01641)*normdis.^3+(-0.1231)*normdis.^2+0.3322*normdis+0.009893;
%%  若出现第i块只有一个数据点的时候，记误差为0
       if isempty(phi{i,1})
           phi{i,1}=0;  
       end
     cluster_cost=cluster_cost+p.*phi{i,1}; %每一块距离代价分别相加
     purity_cost=purity_cost+(p-Max_num);%%每一块纯度代价分别相加
end 
cost1(t,1)=2*cluster_cost;%聚类为K时的总代价
cost2(t,1)=2*purity_cost;
cost(t,1)=0.3*cost1(t,1)+0.7*cost2(t,1)+K;
%cost(t,1)=cost2(t,1)+K;
t=t+1;
end
%costK=cost;
idk=find(cost==min(cost))+1;%找到最小代价的聚类块数K
K=idk;
% Q=zeros(K,c-1);
class=cell(K,1);
%class_diatence=cell(K,1);
label=cell(K,1);
%% 当代价为最小时的聚类块数K作为数据集的聚类簇数
% for i=1:K
%    % C(i,:)=X(randi(m,1),:);     %随机初始化聚类中心%randi(n,1)产生一个1到n的伪随机整数
%    %b=randi(n,1);
%    b=floor((n.*i)./(K+1));
%    Q(i,:)=X(b,:);
% %    buyquery(g,1)=b;
% %    g=g+1;
% end

matrix=Center2{K-1,1};
[idx,C]= kmeans(X,K,'start',matrix);
% [idx,C]= kmeans(X,K);
Center=C;
%label_center=zeros(K,1);
for i=1:K
    class{i,1}=data(idx==i,:);
    label{i,1}=find(idx==i);%%找到第i簇的标号
 
     
end 
end
% cost2=2*cluster_cost+K;