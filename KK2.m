function [label,class,K,Center]=KK2(data,AA,Max)
%% %%����������ͬKK1%%
%% %%ֻ�ǵ����Ĵ��ۺ����ı���%%%%
[n,c]=size(data); %���������ݵ�ĸ���m
%data1=data;
X=data(:,1:c-1);
maxK=floor(sqrt(n));%ȷ�����������������Ϊ10%�����ܸ���
cost=zeros(maxK-1,1);
cost1=zeros(maxK-1,1);%����
cost2=zeros(maxK-1,1);
%% �����ݼ���ֱ��
%  Dists = pdist(X,'euclidean');
%  Max=max(max(Dists));
Center2=cell(maxK-1,1);%���ĵ�
t=1;
%% �ҵ�������������С��Kֵ
for K=2:floor(sqrt(n))
    maxdiameter=cell(K,1);
    %Q=zeros(K,c-1);
    class_distence=cell(K,1);
    class=cell(K,1);
    A=cell(K,1);
    %% �ȼ��ѡȡ��ʼ�������ĵ�
% for i=1:K
%    % C(i,:)=X(randi(m,1),:);     %�����ʼ����������%randi(n,1)����һ��1��n��α�������
%    %b=randi(n,1);
%    b=floor((n.*i)./(K+1));% ѡȡ���ĵ�ı�ţ�����100��۳�2�࣬��ѡȡ��33��66��������Ϊ��ʼ���ĵ㣩
%    Q(i,:)=X(b,:);
% end
% matrix=Q;%��ʼ���ľ���
% [idx,C]= kmeans(X,K,'start',matrix);%�������Դ���Kmeans�㷨�����нϿ�һЩ
%Center{t,1}=C;
[idx,C]= kmeans(X,K);
Center2{t,1}=C;
cluster_cost=0;%ÿһ��Ĵ��۳�ʼֵ��Ϊ0
purity_cost=0;
%% ����������ΪKʱ���ܴ���
for i=1:K
    class{i,1}=data(idx==i,:);%���ڵ�i�������
    A{i,1}=AA(idx==i,:);
    A{i,1}=A{i,1}(:,end);
    b=1:max(A{i,1});
    Max_num=max(histc(A{i,1},b));%%�ҵ������Ѿ�Ԥ�⣬�����δ���������ͬ��ǩ���������ֵ
    [p,~]=size(class{i,1});
    %% �����i���ֱ�����������Ӧ�����
    class_distence{i,1}=X(idx==i,:);
     Dists = pdist(class_distence{i,1},'euclidean');
     maxdiameter{i,1}=max(max(Dists));
     [phi]=badblock(maxdiameter,Max,K);
%       normdis=maxdiameter/Max;
%      phi=(-0.01641)*normdis.^3+(-0.1231)*normdis.^2+0.3322*normdis+0.009893;
%%  �����ֵ�i��ֻ��һ�����ݵ��ʱ�򣬼����Ϊ0
       if isempty(phi{i,1})
           phi{i,1}=0;  
       end
     cluster_cost=cluster_cost+p.*phi{i,1}; %ÿһ�������۷ֱ����
     purity_cost=purity_cost+(p-Max_num);%%ÿһ�鴿�ȴ��۷ֱ����
end 
cost1(t,1)=2*cluster_cost;%����ΪKʱ���ܴ���
cost2(t,1)=2*purity_cost;
cost(t,1)=0.3*cost1(t,1)+0.7*cost2(t,1)+K;
%cost(t,1)=cost2(t,1)+K;
t=t+1;
end
%costK=cost;
idk=find(cost==min(cost))+1;%�ҵ���С���۵ľ������K
K=idk;
% Q=zeros(K,c-1);
class=cell(K,1);
%class_diatence=cell(K,1);
label=cell(K,1);
%% ������Ϊ��Сʱ�ľ������K��Ϊ���ݼ��ľ������
% for i=1:K
%    % C(i,:)=X(randi(m,1),:);     %�����ʼ����������%randi(n,1)����һ��1��n��α�������
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
    label{i,1}=find(idx==i);%%�ҵ���i�صı��
 
     
end 
end
% cost2=2*cluster_cost+K;