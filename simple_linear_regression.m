% Simple Linear Regression

% The dataset consists of CDC growth statistics for boys
%
% x= age (years)
% y= height (meters)
%

clear all; close all; clc;
x = load('linear_regressionx.dat'); y = load('linear_regressiony.dat');

m = length(y); % number of data points = 50

% Visualize the training data
figure;
plot(x, y, 'o');
title({'Training Data'})
ylabel('Y: Height (meters)')
xlabel('X: Age (years)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% 1. Using 'fitlm' function: "fitlm(x,y)" uses intercept by default 2 
a1 = fitlm(x,y);
W=a1.Coefficients{:,1};

 
% Visualize results
figure
hold on
scatter(x,y)
plot(x, W(1)+W(2)*x, '-')
ylabel('Y: Height (meters)')
xlabel('X: Age (years)')
title({'Simple Linear Regression using fitlm(x,y) function'});
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% 2. Using 'regress' function: "regress(y,x)" uses no intercept by default. 

% Add intercept term to x
X = [ones(m, 1), x]; 

a2 = regress(y,X);

% Visualize results
figure
hold on
scatter(x,y)
plot(x, a2(1)+a2(2)*x, '-')
ylabel('Y: Height (meters)')
xlabel('X: Age (years)')
title({'Simple Linear Regression using regress(x,y) function'});
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% 3. Using Stochastic Gradient Descent

% initialize fitting parameters
theta = zeros(2, 1); 
grad = zeros(2,1);
iterations = 1500;
alpha = 0.07;

% SGD iterative updates 
for num_iterations = 1:iterations
   h = X*theta;    %calculate hypothesis function 
      
   grad = (1/m)*(X.'* (h-y));    %calculate gradient 
   theta = theta -alpha*grad;    %update theta
          
end

% Visualize results
figure
hold on
scatter(x,y)
plot(X(:,2), X*theta, '-')
ylabel('Y: Height (meters)')
xlabel('X: Age (years)')
title({'Simple Linear Regression using SGD method'});
hold off