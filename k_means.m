% K-means clustering

% The dataset consists of 150 datapoints of 2 features from iris dataset
%

clear all; close all; clc


X=load('kmeans.dat'); 
K=3; % number of centroids
n=size(X); % check the data size
max_iterations = 10; % if this number is too low a warning will be displayed stating that the algorithm did not converge, which you should expect since the software only implemented one iteration.


figure;
plot(X(:,1),X(:,2),'k.','MarkerSize',12);
title 'Iris Data';
xlabel 'Petal Lengths (cm)'; 
ylabel 'Petal Widths (cm)';

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 1. Using 'kmeans' function: "kmeans(x,k)" performs k-means clustering to 
% partition the observations of the n-by-p data matrix X into k clusters, 
% and returns an n-by-1 vector (idx) containing cluster indices of each observation. 
% Rows of X correspond to points and columns correspond to variables. 

rng(1); % For reproducibility
[idx,C] = kmeans(X,K);

x1 = min(X(:,1)):0.01:max(X(:,1));
x2 = min(X(:,2)):0.01:max(X(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

idx2Region = kmeans(XGrid,K,'MaxIter',max_iterations,'Start',C); 

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)
if K==4
    plot(X(idx==4,1),X(idx==4,2),'y.','MarkerSize',12)
elseif K==5
    plot(X(idx==4,1),X(idx==4,2),'y.','MarkerSize',12)
    plot(X(idx==5,1),X(idx==5,2),'m.','MarkerSize',12)
end
plot(C(:,1),C(:,2),'k*','MarkerSize',15,'LineWidth',3) 
if K==3
    legend('Cluster 1','Cluster 2','Cluster 3','Centroids','Location','NW')
elseif K==4
    legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4', 'Centroids','Location','NW')
elseif K==5 
    legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5', 'Centroids','Location','NW')
end
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)';
title 'Cluster Assignments and Centroids using built-in function'
hold off

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 2. Using your own function

%Initialiaze centroids
if K==3
    rand_centroid_idx = [1,55,125];
elseif K==4 
    rand_centroid_idx = [1,40,80,125];
elseif K==5 
    rand_centroid_idx = [1,35,70,100,125];
end
centroids = X(rand_centroid_idx,:);

%Initialize the metrics used to update centroids
cluster_sum = zeros(K,2);
cluster_samples = zeros(K,1);

for i=1:max_iterations                          %run max_iterations number of iterations
     for num_sample = 1:length(X)               %assign each point to a cluster 

        %Find the eulidean distance between a sample and the centroids
        temp = X(num_sample,:) - centroids;
        euc_dist = vecnorm(temp,2,2);

        % Get the index of centroid with the minimum distance.
        % That will be the cluster we assign the sample to.
        closest_cluster = find(euc_dist == min(euc_dist));

        % Assign the sample to the cluster.
        cluster_assigned(num_sample) = closest_cluster;

        % Add the samples that belong to the same cluster.
        cluster_sum(closest_cluster,:) = cluster_sum(closest_cluster,:) + X(num_sample,:);

        % Track the number of samples in the cluster.
        cluster_samples(closest_cluster) = cluster_samples(closest_cluster)+1;

     end
    % Update the centroids by the mean of all the points that belong to
    % that cluster.
    centroids =  cluster_sum ./ cluster_samples;

    % Reset to zeros for the next iteration
    cluster_sum = zeros(K,2);
    cluster_samples = zeros(K,1);
end 

figure;
plot(X(cluster_assigned==1,1),X(cluster_assigned==1,2),'r.','MarkerSize',12)
hold on
plot(X(cluster_assigned==2,1),X(cluster_assigned==2,2),'b.','MarkerSize',12)
plot(X(cluster_assigned==3,1),X(cluster_assigned==3,2),'g.','MarkerSize',12)
if K==4
    plot(X(cluster_assigned==4,1),X(cluster_assigned==4,2),'y.','MarkerSize',12)
elseif K==5
    plot(X(cluster_assigned==4,1),X(cluster_assigned==4,2),'y.','MarkerSize',12)
    plot(X(cluster_assigned==5,1),X(cluster_assigned==5,2),'m.','MarkerSize',12)
end
plot(centroids(:,1),centroids(:,2),'k*','MarkerSize',15,'LineWidth',3) 
if K==3
    legend('Cluster 1','Cluster 2','Cluster 3','Centroids','Location','NW')
elseif K==4
    legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4', 'Centroids','Location','NW')
elseif K==5 
    legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5', 'Centroids','Location','NW')
end
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)';
title 'Cluster Assignments and Centroids using your implementation'
hold off

