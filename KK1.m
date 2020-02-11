function [class,label,K,Center]=KK1(data,Max,cluster_ratio)
%% 函数的作用是根据第一步的代价来找到最佳聚类块数
%% %%%%%%%%输入%%%%%%%%%%%%%%%%%%%%%%%
%%data:原始数据集数据
%%Max:数据集最远的两点之间的距离，即数据集的直径
[n,c]=size(data); %求输入数据点的个数m
%data1=data;
X=data(:,1:c-1);
maxK=floor(cluster_ratio*n);%确定迭代的最大聚类块数为10%数据总个数(根据数据集大小酌情选择最大聚类块数，可设为0.1*n,0.05*n,sqrt(n))
cost=zeros(maxK-1,1);%代价
%% 求数据集的直径
%  Dists = pdist(X,'euclidean');
%  Max=max(max(Dists));
 Center1=cell(maxK-1,1);%中心点
 t=1;
%% 找到聚类误差代价最小的K值
for K=2:floor(cluster_ratio*n)%%%与maxK保持一致,即是聚类簇数上限
    maxdiameter=cell(K,1);
    %Q=zeros(K,c-1);
    class_distence=cell(K,1);
    class=cell(K,1);
    %% 等间隔选取初始聚类中心点
% for i=1:K
%     %C(i,:)=X(randi(m,1),:);     %随机初始化聚类中心%randi(n,1)产生一个1到n的伪随机整数
%     %b=randi(n,1);
%    b=floor((n.*i)./(K+1));% 选取初始中心点的标号（例如100点聚成2类，就选取了33和66两个点作为初始中心点）
%    Q(i,:)=X(b,:);
% end
% matrix=Q;%初始中心矩阵
% [idx,C]= kmeans(X,K,'start',matrix);%调用了自带的Kmeans算法，运行较快一些
%Center{t,1}=C;
[idx,C]= kmeans(X,K);
Center1{t,1}=C;
cluster_cost=0;%每一块的代价初始值赋为0
%% 计算聚类块数为K时的总代价
for i=1:K
    class{i,1}=data(idx==i,:);%属于第i块的数据
    [p,~]=size(class{i,1});
    %% 计算第i块的直径，并求出相应的误差
    class_distence{i,1}=X(idx==i,:);
     Dists = pdist(class_distence{i,1},'euclidean');
     maxdiameter{i,1}=max(max(Dists));
     [phi]=badblock(maxdiameter,Max,K);%%误差拟合函数
%       normdis=maxdiameter/Max;
%      phi=(-0.01641)*normdis.^3+(-0.1231)*normdis.^2+0.3322*normdis+0.009893;
%%  若出现第i块只有一个数据点的时候，记误差为0
       if isempty(phi{i,1})
           phi{i,1}=0;  
       end
     cluster_cost=cluster_cost+p.*phi{i,1}; %每一块代价分别相加
end 
cost(t,1)=2*cluster_cost+K;%聚类为K时的总代价
t=t+1;
end
%costK=cost;
idk=find(cost==min(cost))+1;%找到最小代价的聚类块数K
K=idk;
% Q=zeros(K,c-1);
class=cell(K,1);
%class_diatence=cell(K,1);
label=cell(K,1);
% Center=cell(K,1);
% t=1;
%% 当代价为最小时的聚类块数K作为数据集的聚类簇数
% for i=1:K
%    % C(i,:)=X(randi(m,1),:);     %随机初始化聚类中心%randi(n,1)产生一个1到n的伪随机整数
%    %b=randi(n,1);
%    b=floor((n.*i)./(K+1));
%    Q(i,:)=X(b,:);
% %    buyquery(g,1)=b;
% %    g=g+1;
% end
matrix=Center1{K-1,1};
[idx,C]= kmeans(X,K,'start',matrix);
% [idx,C]= kmeans(X,K);
 Center=C;
% t=t+1;
%cluster_cost=0;
%label_center=zeros(K,1);
for i=1:K
    class{i,1}=data(idx==i,:);
    label{i,1}=find(idx==i);%%找到第i簇的标号
    
end 
end
