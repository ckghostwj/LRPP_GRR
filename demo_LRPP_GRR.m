% % % The code is written by Jie Wen, if you have any problems, 
% % % please don't hesitate to contact me: wenjie@hrbeu.edu.cn 
% % % run "demo.LRPP_GRR.m"  to implement the code
% % % 
% % % If you find the code is useful, please cite the following reference:
% % % Wen J, Han N, Fang X, et al. Low-Rank Preserving Projection Via Graph Regularized Reconstruction[J]. 
% % % IEEE Transactions on Cybernetics, 2018. doi: 10.1109/TCYB.2018.2799862 
clc,clear;
clear all;
clear memory;
name = 'YaleB_32x32_98';
load(name)

nnClass = length(unique(gnd));
num_Class = [];
for i = 1:nnClass
    num_Class = [num_Class length(find(gnd==i))]; 
end
sele_num  = 10; 
Train_Ma  = [];
Train_Lab = [];
Test_Ma   = [];
Test_Lab  = [];
for j =1:nnClass
    idx = find(gnd==j);
    randIdx = randperm(num_Class(j));
    Train_Ma  = [Train_Ma; fea(idx(randIdx(1:sele_num)),:)];
    Train_Lab = [Train_Lab;gnd(idx(randIdx(1:sele_num)))];
    Test_Ma   = [Test_Ma;fea(idx(randIdx(sele_num+1:num_Class(j))),:)];
    Test_Lab  = [Test_Lab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];
end
Train_Ma = Train_Ma';
Train_Ma = Train_Ma./repmat(sqrt(sum(Train_Ma.^2)),[size(Train_Ma,1) 1]);
Test_Ma  = Test_Ma';
Test_Ma  = Test_Ma./repmat(sqrt(sum(Test_Ma.^2)),[size(Test_Ma,1) 1]);
best_ac  = 0;
lambda1  = 1e-2;
lambda2  = 1e-4;
dim = 140;                                                                                                                                                                                                           00;

X = Train_Ma;
options = [];
options.ReducedDim = dim;
[P1,~] = PCA1(X', options);
% ------------- 运行出错的问题主要在这里 ------------------ % 样本数不足时容易出问题
if size(P1,2) < dim
    P1(:,size(P1,2)+1:dim) = 0;
end
options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'Binary';
W = constructW(X',options);

mu = 0.1;
rho = 1.01;
Max_iter = 80;
[P,Q,Z,obj] = LRPP_GRR(X,P1,W,lambda1,lambda2,dim,mu,rho,Max_iter);
train_data = Q'*Train_Ma;
test_data  = Q'*Test_Ma;
train_data = train_data./repmat(sqrt(sum(train_data.^2)),[size(train_data,1) 1]);
test_data  = test_data./repmat(sqrt(sum(test_data.^2)),[size(test_data,1) 1]);
[class_test] = knnclassify(test_data',train_data',Train_Lab,1,'euclidean','nearest');
rate   = sum(Test_Lab == class_test)/length(Test_Lab)*100
