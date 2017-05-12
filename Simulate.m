%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation and Estimation Code: Mingsheng Life Insurance
% 
% Code written by Beomjoon Shim
% Data: 10/28/2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bGetSolution = true;

if bGetSolution == true
%     clear all
    clc
end 

delta = 0.95;
nSim = 100;

% State variables
nQ = 3; % Month type in a quarter
Interval_S = 2000;
Interval_S_2m = 2000; % Interval for sales
Interval_S_t = 10000; % Interval for team sales

N_it1 = [0:5]'; % A salesperson can have at most 3 level 1 salesperson
N_it2 = [0:5]'; % A salesperson can have at most 3 level 2 salesperson
nN_it1 = length(N_it1); % Number of salespeople in each level
nN_it2 = length(N_it2); % Number of salespeople in each level

S = [0:4]'*Interval_S;
S_2m = [0:4]'*Interval_S_2m; % Individual sales
S_t = [0:3]'*Interval_S_t; % Team sales

% Create Team(N1,N2) state variable
N_it = zeros(1);
index1 = 1;
index2 = 1;
for i=1:nN_it2
    for j=1:nN_it1
        N_it(index1,1) = N_it1(j);
        N_it(index1,2) = N_it2(i);
        index1 = index1 + 1;
        index2 = index2 + 1;
    end
end
nN_it = length(N_it);

% Create Sales state variable (including S_2m & s_it-1)
Sales = zeros(1);
row = 1;
for i=0:max(S_2m)/Interval_S_2m
    for j=0:i
        Sales(row,1) = i*Interval_S_2m;
        Sales(row,2) = j*Interval_S;
        row = row+1;
    end
end

% Create Team-level Sales state variable
Sales_t = zeros(1);
row = 1;
for i=0:max(S_t)/Interval_S_t
    for j=0:i
        Sales_t(row,1) = i*Interval_S_t;
        Sales_t(row,2) = j*Interval_S_t;
        row = row+1;
    end
end

nS = length(S);
nS_2m = length(S_2m);
nS_t = length(S_t);
nSales = length(Sales);
nSales_t = length(Sales_t);

Effort_interval = 0.2;
nEffort1 = (1/Effort_interval)+1;
index1 = 1;
index2 = 1;
Effort = 0;
for i=1:nEffort1
    for j=1:nEffort1
        Effort(index1,1) = (j-1)*Effort_interval;
        Effort(index1,2) = (index2-1)*Effort_interval;
        index1 = index1 + 1;
    end
    index2 = index2 + 1;
end
nEffort = length(Effort);

% Set the model parameters
beta = 150;
alpha = 2500;
tau1 = 1000; % each level 1 salesperson's productivity
tau2 = 1500; % each level 2 salesperson's productivity
theta1 = 800;
theta2 = 400;
theta3 = 300;
mu = 90;
p_1stay = 0.95;
p_2stay = 0.95;

% Generate ramdom components
e_a_1 = randn(nSim,nQ)*1000;
e_a_2 = randn(nSim,nQ)*1000;
e_b = randn(nSim,nQ)*30;
e_t1_1 = randn(nSim,nQ)*500;
e_t1_2 = randn(nSim,nQ)*500;
e_t2 = randn(nSim,nQ)*1000;
e_stay1_1 = zeros(nN_it1,nSim,nQ); % level1
e_stay1_2 = zeros(nN_it1,nSim,nQ); % level2
for i=1:nN_it1
    e_stay1_1(i,:,:) = binornd(N_it1(i),p_1stay,nSim,nQ);
    e_stay1_2(i,:,:) = binornd(N_it1(i),p_1stay,nSim,nQ);
end
e_stay2_1 = zeros(nN_it2,nSim,nQ); % level1
e_stay2_2 = zeros(nN_it2,nSim,nQ); % level2
for i=1:nN_it2
    for j=1:nQ
        if j == 3
            e_stay2_1(i,:,j) = binornd(N_it2(i),p_2stay,nSim,1);
            e_stay2_2(i,:,j) = binornd(N_it2(i),p_2stay,nSim,1);
        else
            e_stay2_1(i,:,j) = (i-1)*ones(1,nSim);
            e_stay2_2(i,:,j) = (i-1)*ones(1,nSim);
        end
    end
end

save e_a_1.mat e_a_1
save e_a_2.mat e_a_2
save e_b.mat e_b
save e_t1_1.mat e_t1_1
save e_t1_2.mat e_t1_2
save e_t2.mat e_t2
save e_stay1_1.mat e_stay1_1
save e_stay1_2.mat e_stay1_2
save e_stay2_1.mat e_stay2_1
save e_stay2_2.mat e_stay2_2

load e_a_1.mat e_a_1
load e_a_2.mat e_a_2
load e_b.mat e_b
load e_t1_1.mat e_t1_1
load e_t1_2.mat e_t1_2
load e_t2.mat e_t2
load e_stay1_1.mat e_stay1_1
load e_stay1_2.mat e_stay1_2
load e_stay2_1.mat e_stay2_1
load e_stay2_2.mat e_stay2_2

% Anonymous functions for converting discrete state space variables...
toS = @(x) min(max(round(x/Interval_S)+1,1),nS); % calculating S
toS_2m = @(x) min(max(round(x/Interval_S_2m)+1,1),nS_2m); % calculating S_2m
toS_t = @(x) min(max(round(x/Interval_S_t)+1,1),nS_t); % calculating S_t
toA = @(x) min(max(floor(x/mu),0),max(N_it1));
toN = @(x) min(max(round(x),0),max(N_it1));
% toN = @(x) min(max(round(x)+1,1),length(N_it1)); % make x as index..

% Create indexes to speed up the computation
N_it2index = zeros(nN_it1,nN_it2);
for i=1:nN_it
    N_it2index(N_it(i,1)+1,N_it(i,2)+1) = i;
end

Sales2index = zeros(nS_2m,nS_2m);
for i=1:nSales
    Sales2index(Sales(i,1)/Interval_S_2m+1,Sales(i,2)/Interval_S+1) = i;
end

Sales_t2index = zeros(nS_t,nS_t);
for i=1:nSales_t
    Sales_t2index(Sales_t(i,1)/Interval_S_t+1,Sales_t(i,2)/Interval_S_t+1) = i;
end

% Do the contraction mapping...
diff = 1;
tol = 1e-6; % Set tolerance level for contraction mapping: 10^-6

nIter = 1;

V1_old = zeros(nN_it,nSales,nSales_t,nQ);
V1_new = zeros(nN_it,nSales,nSales_t,nQ);
V2_old = zeros(nN_it,nSales,nSales_t,nQ);
V2_new = zeros(nN_it,nSales,nSales_t,nQ);

effort1 = zeros(nN_it,nSales,nSales_t,nQ);
effort2 = zeros(nN_it,nSales,nSales_t,nQ);

if bGetSolution == true
    % profile on
    while(abs(diff) > tol)
        % Vectorized code..
        tic
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Level 1 salesperson
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Individual sales
        s_it = bsxfun(@plus,alpha*Effort(1:nEffort1,1),reshape(e_a_1,1,nSim,1,nQ));% 6*50*1*3
        S_3m_it = bsxfun(@plus,reshape(Sales(:,1),1,1,nSales),s_it); % 6*50*15*3

        % Bonus calculation for lvl 1 salesperson
%         temp = 0.5 * S_3m_it(:,:,:,3);
        temp = 0.3 * S_3m_it(:,:,:,3);
        b_it = zeros(size(S_3m_it));
        b_it(:,:,:,3) = (temp >= 4000 & temp < 6000).*temp*0.22+(temp >= 6000 & temp < 12000).*temp*0.25+...
            (temp >= 12000 & temp < 24000).*temp*0.3+(temp >= 24000 & temp < 48000).*temp*0.35+...
            (temp >= 48000).*temp*0.45;

        % Calculate next period n_it
        n_it1 = toN(bsxfun(@plus,e_stay1_1(N_it(:,1)+1,:,:)-e_stay2_1(N_it(:,2)+1,:,:),N_it(:,2)));
        n_it2 = toN(bsxfun(@plus,e_stay2_1(N_it(:,2)+1,:,:)-e_stay1_1(N_it(:,1)+1,:,:),N_it(:,1)));

        % Team sales for lvl 1 salesperson
        s_t = bsxfun(@plus,tau1*N_it(:,1),bsxfun(@times,N_it(:,1),reshape(e_t1_1,1,nSim,nQ))); % 16*50*3
    %     s_t = zeros(size(s_t)); % test code

        u_ind = bsxfun(@plus,0.3*s_it,b_it); % 6*50*15*3
        u_team = 0.1*0.3*s_t;
        cost = theta1*(Effort(1:nEffort1,1).^2);
        temp = bsxfun(@plus,reshape(u_ind,nEffort1,1,nSim,nSales,nQ),reshape(u_team,1,nN_it,nSim,1,nQ));
        u_it = bsxfun(@minus,temp,cost);

        % Check promotion condition for lvl 1 salesperson
        IsPromo = reshape(0.3*S_3m_it >= 2000,nEffort1,1,nSim,nSales,1,nQ);

        temp1 = N_it2index(sub2ind(size(N_it2index),reshape(n_it1,nN_it*nSim*nQ,1)+1,reshape(n_it2,nN_it*nSim*nQ,1)+1));
        arg1 = repmat(reshape(temp1,1,nN_it,nSim,1,1,nQ),[nEffort1,1,1,nSales,nSales_t,1]);

        temp = toS_2m(bsxfun(@plus,reshape(Sales(:,2),1,1,nSales),s_it)); % 6*50*15*3
        temp1 = reshape(temp,nEffort1*nSim*nSales*nQ,1);
        temp2 = reshape(repmat(toS(s_it),[1 1 nSales 1]),nEffort1*nSim*nSales*nQ,1);
        temp = Sales2index(sub2ind(size(Sales2index),temp1,temp2)); % 13500*1
        arg2 = repmat(reshape(temp,nEffort1,1,nSim,nSales,1,nQ),[1,nN_it,1,1,nSales_t,1]);   

        temp = toS_t(bsxfun(@plus,reshape(Sales_t(:,2),1,1,nSales_t),reshape(s_t,nN_it,nSim,1,nQ)));
        temp1 = reshape(temp,nN_it*nSim*nSales_t*nQ,1);
        temp2 = reshape(repmat(reshape(toS_t(s_t),nN_it,nSim,1,nQ),1,1,nSales_t,1),nN_it*nSim*nSales_t*nQ,1);
        temp = Sales_t2index(sub2ind(size(Sales_t2index),temp1,temp2)); % this is 25 times faster...
        arg3 = repmat(reshape(temp,1,nN_it,nSim,1,nSales_t,nQ),[nEffort1 1 1 nSales 1 1]);

        % Integration..
        u_it = reshape(sum(u_it,3)/nSim,nEffort1,nN_it,nSales,1,nQ);
        temp1 = zeros(nEffort1,nN_it,nSales,nSales_t,nQ);
        temp2 = zeros(nEffort1,nN_it,nSales,nSales_t,nQ);
        for t=1:3    
            tmp1 = V1_old(sub2ind(size(V1_old),arg1(:,:,:,:,:,t),arg2(:,:,:,:,:,t),arg3(:,:,:,:,:,t),repmat(toQ(t+1),nEffort1,nN_it,nSim,nSales,nSales_t)));
            tmp1 = bsxfun(@times,~IsPromo(:,:,:,:,:,t),tmp1);
            tmp2 = V2_old(sub2ind(size(V2_old),arg1(:,:,:,:,:,t),arg2(:,:,:,:,:,t),arg3(:,:,:,:,:,t),repmat(toQ(t+1),nEffort1,nN_it,nSim,nSales,nSales_t)));
            tmp2 = bsxfun(@times,IsPromo(:,:,:,:,:,t),tmp2);
            temp1(:,:,:,:,t) = reshape(sum(tmp1,3)/nSim,nEffort1,nN_it,nSales,nSales_t);
            temp2(:,:,:,:,t) = reshape(sum(tmp2,3)/nSim,nEffort1,nN_it,nSales,nSales_t);
        end

    %       [out1, out2] = max(bsxfun(@plus,u_it,delta*(temp1)),[],1);
        [out1, out2] = max(bsxfun(@plus,u_it,delta*(temp1+temp2)),[],1);

        V1_new = reshape(out1,nN_it,nSales,nSales_t,nQ);
        effort1 = reshape(out2,nN_it,nSales,nSales_t,nQ);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Level 2 salesperson 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Individual sales
        s_it = bsxfun(@plus,alpha*Effort(:,1),reshape(e_a_2,1,nSim,1,nQ));% 36*50*1*3
        S_3m_it = bsxfun(@plus,reshape(Sales(:,1),1,1,nSales),s_it); % 36*50*15*3

        % Bonus calculation for lvl 1 salesperson
        temp = 0.3 * S_3m_it(:,:,:,3);
        b_it = zeros(size(S_3m_it));
        b_it(:,:,:,3) = (temp >= 4000 & temp < 6000).*temp*0.22+(temp >= 6000 & temp < 12000).*temp*0.25+...
            (temp >= 12000 & temp < 24000).*temp*0.3+(temp >= 24000 & temp < 48000).*temp*0.35+...
            (temp >= 48000).*temp*0.45;

        % Adding salesmen for lvl 2 salesperson
        n_add = toA(bsxfun(@plus,beta*Effort(:,2),reshape(e_b,1,1,nSim,1,nQ)));

        % Calculate next period n_it
        temp = bsxfun(@plus,e_stay1_2(N_it(:,1)+1,:,:)-e_stay2_2(N_it(:,2)+1,:,:),N_it(:,2));
        n_it1 =  toN(bsxfun(@plus,n_add,reshape(temp,1,nN_it,nSim,1,nQ)));
        temp = toN(bsxfun(@plus,e_stay2_2(N_it(:,2)+1,:,:)-e_stay1_2(N_it(:,1)+1,:,:),N_it(:,1)));
        n_it2 = repmat(reshape(temp,1,nN_it,nSim,1,nQ),nEffort,1,1,1);

        % Team sales for lvl 2 salesperson
        s_t = bsxfun(@plus,tau1*N_it(:,1)+tau2*N_it(:,2),bsxfun(@times,N_it(:,1),reshape(e_t1_2,1,nSim,1,nQ))+bsxfun(@times,N_it(:,2),reshape(e_t2,1,nSim,1,nQ))); % 16*50*3

        u_ind = bsxfun(@plus,0.3*s_it,b_it); % 36*50*15*3
        u_team = 0.1*0.3*s_t;
        cost = theta1*(Effort(:,1).^2)+theta2*(Effort(:,2).^2)+theta3*Effort(:,1).*Effort(:,2);
        temp = bsxfun(@plus,reshape(u_ind,nEffort,1,nSim,nSales,nQ),reshape(u_team,1,nN_it,nSim,1,nQ));
        u_it = bsxfun(@minus,temp,cost);

        % Check demotion condition for lvl 2 salesperson
        IsDemo = zeros(nEffort,1,nSim,nSales,1,nQ);
        IsDemo(:,:,:,:,:,3) = reshape(0.3*S_3m_it(:,:,:,3) < 600,nEffort,1,nSim,nSales,1,1);


        temp1 = N_it2index(sub2ind(size(N_it2index),n_it1+1,n_it2+1));
        arg1 = repmat(reshape(temp1,nEffort,nN_it,nSim,1,1,nQ),[1,1,1,nSales,nSales_t,1]);

        temp1 = toS_2m(bsxfun(@plus,reshape(Sales(:,2),1,1,nSales),s_it)); % 36*50*15*3
        temp2 = repmat(toS(s_it),[1 1 nSales 1]);
        temp = Sales2index(sub2ind(size(Sales2index),temp1,temp2)); % 81000*1
        arg2 = repmat(reshape(temp,nEffort,1,nSim,nSales,1,nQ),[1,nN_it,1,1,nSales_t,1]);   

        temp1 = toS_t(bsxfun(@plus,reshape(Sales_t(:,2),1,1,nSales_t),s_t));
        temp2 = repmat(toS_t(s_t),1,1,nSales_t);
        temp = Sales_t2index(sub2ind(size(Sales_t2index),temp1,temp2)); % this is 25 times faster...
        arg3 = repmat(reshape(temp,1,nN_it,nSim,1,nSales_t,nQ),[nEffort 1 1 nSales 1 1]);

        % Integration..
        u_it = reshape(sum(u_it,3)/nSim,nEffort,nN_it,nSales,1,nQ);
        temp1 = zeros(nEffort,nN_it,nSales,nSales_t,nQ);
        temp2 = zeros(nEffort,nN_it,nSales,nSales_t,nQ);
        for t=1:3
            if t == 3
                tmp1 = V1_old(sub2ind(size(V1_old),arg1(:,:,:,:,:,t),arg2(:,:,:,:,:,t),arg3(:,:,:,:,:,t),repmat(toQ(t+1),nEffort,nN_it,nSim,nSales,nSales_t)));
                tmp1 = bsxfun(@times,IsDemo(:,:,:,:,:,t),tmp1);
                temp1(:,:,:,:,t) = reshape(sum(tmp1,3)/nSim,nEffort,nN_it,nSales,nSales_t);
            end
            tmp2 = V2_old(sub2ind(size(V2_old),arg1(:,:,:,:,:,t),arg2(:,:,:,:,:,t),arg3(:,:,:,:,:,t),repmat(toQ(t+1),nEffort,nN_it,nSim,nSales,nSales_t)));
            tmp2 = bsxfun(@times,~IsDemo(:,:,:,:,:,t),tmp2);
            temp2(:,:,:,:,t) = reshape(sum(tmp2,3)/nSim,nEffort,nN_it,nSales,nSales_t);
        end

        [out1, out2] = max(bsxfun(@plus,u_it,delta*(temp1+temp2)),[],1);

        V2_new = reshape(out1,nN_it,nSales,nSales_t,nQ);
        effort2 = reshape(out2,nN_it,nSales,nSales_t,nQ);

        diff = max(reshape(abs(V1_new-V1_old),nN_it*nSales*nSales_t*nQ,1))+max(reshape(abs(V2_new-V2_old),nN_it*nSales*nSales_t*nQ,1));

        V1_old = V1_new;
        V2_old = V2_new;

        nIter = nIter + 1;
        disp(diff);
        toc
    end
    % temp = Effort(1:nEffort1,1); % needs to check...
    temp = (effort1*Effort_interval-Effort_interval).*(effort1>0)

    aa1 = Effort(:,1);
    aa2 = Effort(:,2);

    disp(nIter);

    % profile report;
    % profile off;

    save V1_new.mat V1_new
    save V2_new.mat V2_new
    save effort1.mat effort1
    save effort2.mat effort2
end


if bGetSolution == false
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Simulation Code Start
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    load V1_new.mat V1_new
    load V2_new.mat V2_new
    load effort1.mat effort1
    load effort2.mat effort2
    
    nSalesman = 5;
    nPeriod = 24;

    % Generate ramdom components
    err_a_1 = randn(nSalesman,nPeriod)*200;
    err_a_2 = randn(nSalesman,nPeriod)*200;
    err_b = randn(nSalesman,nPeriod)*30;
    err_t1_1 = randn(nSalesman,nPeriod)*500;
    err_t1_2 = randn(nSalesman,nPeriod)*500;
    err_t2 = randn(nSalesman,nPeriod)*1000;
    err_stay1_1 = zeros(nN_it1,nSalesman,nPeriod); % level1
    err_stay1_2 = zeros(nN_it1,nSalesman,nPeriod); % level2
    for i=1:nN_it1
        err_stay1_1(i,:,:) = binornd(N_it1(i),p_1stay,nSalesman,nPeriod);
        err_stay1_2(i,:,:) = binornd(N_it1(i),p_1stay,nSalesman,nPeriod);
    end
    err_stay2_1 = zeros(nN_it2,nSalesman,nPeriod); % level1
    err_stay2_2 = zeros(nN_it2,nSalesman,nPeriod); % level2
    for i=1:nN_it2
        for j=1:nPeriod
            if mod(j,3) == 0
                err_stay2_1(i,:,j) = binornd(N_it2(i),p_2stay,nSalesman,1);
                err_stay2_2(i,:,j) = binornd(N_it2(i),p_2stay,nSalesman,1);
            else
                err_stay2_1(i,:,j) = (i-1)*ones(1,nSalesman);
                err_stay2_2(i,:,j) = (i-1)*ones(1,nSalesman);
            end
        end
    end
    
    save err_a_1.mat err_a_1
    save err_a_2.mat err_a_2
    save err_b.mat err_b
    save err_t1_1.mat err_t1_1
    save err_t1_2.mat err_t1_2
    save err_t2.mat err_t2
    save err_stay1_1.mat err_stay1_1
    save err_stay1_2.mat err_stay1_2
    save err_stay2_1.mat err_stay2_1
    save err_stay2_2.mat err_stay2_2
    
    load err_a_1.mat err_a_1
    load err_a_2.mat err_a_2
    load err_b.mat err_b
    load err_t1_1.mat err_t1_1
    load err_t1_2.mat err_t1_2
    load err_t2.mat err_t2
    load err_stay1_1.mat err_stay1_1
    load err_stay1_2.mat err_stay1_2
    load err_stay2_1.mat err_stay2_1
    load err_stay2_2.mat err_stay2_2
    
    Sales_hist = zeros(nSalesman,nPeriod+2);
    TeamSales_hist = zeros(nSalesman,nPeriod+2);
    Team_hist = zeros(nSalesman,nPeriod,2);
    Position_hist = ones(nSalesman,nPeriod);
    Effort_sales = ones(nSalesman,nPeriod)*-1;
    Effort_add = ones(nSalesman,nPeriod)*-1;
    
    for i = 1:nSalesman
        for t = 1:nPeriod
            Sales_2m = sum(Sales_hist(i,t:t+1)')';
            Sales_t_2m = sum(TeamSales_hist(i,t:t+1)')';
            position = Position_hist(i,t);
            
            sales = 0;
            if position == 1
                % Calculate Sales index
                s_index = Sales2index(toS_2m(Sales_2m),toS(Sales_hist(i,t+1)));
                st_index = Sales_t2index(toS_t(Sales_t_2m),toS_t(TeamSales_hist(i,t+1)));
                
                Effort_sales(i,t) = Effort(effort1(N_it2index(Team_hist(i,t,1)+1,Team_hist(i,t,2)+1),s_index,st_index,toQ(t)),1);
                Sales_hist(i,t+2) = alpha*Effort_sales(i,t)+err_a_1(i,t);
                if Sales_hist(i,t+2) < 0
                    Sales_hist(i,t+2) = 0;
                end
                
                % Team evolution
                Team_hist(i,t+1,1) = toN(err_stay1_1(Team_hist(i,t,1)+1,i,t)-err_stay2_1(Team_hist(i,t,2)+1,i,t)+Team_hist(i,t,2));
                Team_hist(i,t+1,2) = toN(err_stay2_1(Team_hist(i,t,2)+1,i,t)-err_stay1_1(Team_hist(i,t,1)+1,i,t)+Team_hist(i,t,1));
                
                TeamSales_hist(i,t+2) = tau1*Team_hist(i,t,1)*err_t1_1(i,t);
                if TeamSales_hist(i,t+2) < 0
                    TeamSales_hist(i,t+2) = 0;
                end
                
                % Check promotion
                if 0.3*(Sales_2m + Sales_hist(i,t+2)) >= 2000
                    Position_hist(i,t+1) =  2;
                end
                
            elseif position == 2
                % Calculate Sales & TeamSales index
                s_index = Sales2index(toS_2m(Sales_2m),toS(Sales_hist(i,t+1)));
                st_index = Sales_t2index(toS_t(Sales_t_2m),toS_t(TeamSales_hist(i,t+1)));
                
                Effort_sales(i,t) = Effort(effort2(N_it2index(Team_hist(i,t,1)+1,Team_hist(i,t,2)+1),s_index,st_index,toQ(t)),1);
                Effort_add(i,t) = Effort(effort2(N_it2index(Team_hist(i,t,1)+1,Team_hist(i,t,2)+1),s_index,st_index,toQ(t)),2);
                
                Sales_hist(i,t+2) = alpha*Effort_sales(i,t)+err_a_2(i,t);
                if Sales_hist(i,t+2) < 0
                    Sales_hist(i,t+2) = 0;
                end
                
                n_add = toA(beta*Effort_add(i,t)+err_b(i,t));
                
                % Team evolution
                Team_hist(i,t+1,1) = toN(err_stay1_1(Team_hist(i,t,1)+1,i,t)-err_stay2_1(Team_hist(i,t,2)+1,i,t)+Team_hist(i,t,2)+n_add);
                Team_hist(i,t+1,2) = toN(err_stay2_1(Team_hist(i,t,2)+1,i,t)-err_stay1_1(Team_hist(i,t,1)+1,i,t)+Team_hist(i,t,1));
                
                TeamSales_hist(i,t+2) = tau1*Team_hist(i,t,1)*err_t1_2(i,t)+tau2*Team_hist(i,t,2)*err_t2(i,t);
                if TeamSales_hist(i,t+2) < 0
                    TeamSales_hist(i,t+2) = 0;
                end
                
                % Check demotion
                if 0.3*(Sales_2m + Sales_hist(i,t+2)) < 600 & toQ(t) == 3
                    Position_hist(i,t+1) =  1;
                else
                    Position_hist(i,t+1) =  2;
                end
                
            end
        end
    end
end
