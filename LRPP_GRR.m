function [P,Q,Z,obj] = LRPP_GRR(X,P1,W,lambda1,lambda2,dim,mu,rho,Max_iter)
% % % The code is written by Jie Wen, if you have any problems, 
% % % please don't hesitate to contact me: wenjie@hrbeu.edu.cn 
% % % run "demo.LRPP_GRR.m"  to implement the code
% % % 
% % % If you find the code is useful, please cite the following reference:
% % % Wen J, Han N, Fang X, et al. Low-Rank Preserving Projection Via Graph Regularized Reconstruction[J]. 
% % % IEEE Transactions on Cybernetics, 2018. doi: 10.1109/TCYB.2018.2799862 
[m,n] = size(X); % (m: dimension; n: number);
max_mu = 10^5;
D = diag(sum(W,2));
%%------------------------------initilzation-------------------------------
Z = W;
U = W;

C1 = zeros(dim,n);
C2 = zeros(n,n);

Q = P1;

Y  = Q'*X*Z;
v = sqrt(sum(Q.*Q,2)+eps);
a = 1./v; 
%%-------------------------------------------------------------------------
P = P1;
for iter = 1:Max_iter
    Z_old = Z;
    U_old = U;
    P_old = P;
    Q_old = Q;
    Y_old = Y;
    % ---------------- Q ---------------- %
    B1 = Y-C1/mu;  
    A = diag(a);
    Q = (lambda2*A+mu*X*Z*Z'*X')\(mu*X*Z*B1');
    v = sqrt(sum(Q.*Q,2)+eps);
    a = 1./v; 
    % ---------- Z  --------------- %
    B1 = Y-C1/mu;
    B2 = U-C2/mu;
    Z = (X'*Q*Q'*X+eye(n))\(X'*Q*B1+B2);
    % ------ U ----- %
    es = lambda1/mu;
    temp_U = Z+C2/mu;
    [uu,ss,vv] = svd(temp_U,'econ');
    ss = diag(ss);
    SVP = length(find(ss>es));
    if SVP>1
        ss = ss(1:SVP)-es;
    else
        SVP = 1;
        ss = 0;
    end
    U = uu(:,1:SVP)*diag(ss)*vv(:,1:SVP)';   
    % ---------- Y ------------- %
    B3 = Q'*X*Z+C1/mu;  
    Y = (2*P'*X*W+mu*B3)*inv(2*D+mu*eye(n));
    % ----------- P ------------- %   
    M = X*W*Y';
    [U1,~,S1] = svd(M,'econ');
    P = U1*S1';    
    % -------- C1;C2;mu ------------- %
    C1 = C1+mu*(Q'*X*Z-Y);
    C2 = C2+mu*(Z-U);
    LL1 = norm(Z-Z_old,'fro');
    LL2 = norm(U-U_old,'fro');
    LL3 = norm(Y-Y_old,'fro');
    LL4 = norm(Q-Q_old,'fro');
    LL5 = norm(P-P_old,'fro');
    SLSL = max(max(max(max(LL1,LL2),LL3),LL4),LL5);
    if SLSL*mu/norm(X,'fro') < 0.01
        mu = min(rho*mu,max_mu);
    end
    
    leq1 = norm(Q'*X*Z-Y,Inf);
    leq2 = norm(Z-U,Inf);
%     obj(iter) = trace(X*D*X'-2*X*W*Y'*P'+P*Y*D*Y'*P')+lambda2*sum(v)+lambda1*rank(Z);
    obj(iter) = (trace(X*D*X'-2*X*W*Y'*P')+trace(Y*D*Y')+lambda2*sum(v)+lambda1*sum(ss))/norm(X,'fro')^2;  
    if iter > 2
        if abs(obj(iter)-obj(iter-1)) < 10^-7
            iter
            break;
        end
    end   
    
end
end
