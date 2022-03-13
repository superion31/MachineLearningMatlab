% Multivariate Linear Regression

% Data includes 47 datapoints of housing prices in Portland, Oregon.
%
% x=(x1,x2)=(area, number of bedrooms)
% y= house prices


clear all; close all; clc

x = load('mv_regressionx.dat'); 
y = load('mv_regressiony.dat');

m = length(y);

% Visualize Data 
scatter3(x(:,1), x(:,2), y)
title({'Training Data'})
ylabel('Number of Bedrooms')
xlabel('Area')
zlabel('House Prices')

% Add intercept term to x
X = [ones(m, 1), x];

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 1. Using 'mvregress' function: "mvregress(x,y)" returns the estimated 
% coefficients for a multivariate normal regression of the d-dimensional 
% responses in Y on the design matrices in X.
beta1=mvregress(X,y)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 2. Using Stochastic Gradient Descent with different alphas

% Scale features and set them to zero mean
mu = mean(x);
sigma = std(x);

x_scaled = (x-mu)./sigma;
X_scaled = [ones(m, 1), x_scaled];

% Save unscaled features to calculate the parameters from the normal equation later
x_unscaled = X;

%create a plot to compare different alpha values

%initialize parameters
figure
iters= 100;                                                  
alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3];
plotstyle = {'b', 'r', 'g', 'k', 'b--', 'r--'}; 

theta_grad_descent = zeros(size(X_scaled(1,:)));   %to store the values of theta of the best learning rate

for alpha_i = 1:length(alpha) 
    theta = zeros(size(X_scaled,2),1); 
    Jtheta = zeros(iters, 1);
    for i = 1:iters
        h = X_scaled * theta;
        Jtheta(i) = (1/(2*m)).*(h-y)'* (h -y);      %cost compute 
        grad = (1/m)*(X_scaled.'* (h-y));           %grad compute
        theta = theta - alpha(alpha_i).*grad;       %theta update
    end

    plot(1:50, Jtheta(1:50),char(plotstyle(alpha_i)),'LineWidth', 2)   %plot cost vs iter_num over alpha values
    hold on

    if(alpha(alpha_i) == 1)                                              %save the best alpha 
        theta_grad_descent = theta ;                                     %examining the plot alpha = 1 has the 
    end                                                                  %best performance
 
end
legend('0.01','0.03','0.1','0.3','1','1.3');
xlabel('Number of iterations')
ylabel('Cost function')

% print gradient descent's result
theta_grad_descent;

% Calculate the parameters from the normal equation
theta_normal = zeros(size(X_scaled(1,:))); 
theta_normal(1) = theta_grad_descent(1) - ((theta_grad_descent(2)*mu(1)/sigma(1) + theta_grad_descent(3)*mu(2)/sigma(2)));
theta_normal(2) = theta_grad_descent(2)/sigma(1);
theta_normal(3) = theta_grad_descent(3)/sigma(2);

theta_normal


