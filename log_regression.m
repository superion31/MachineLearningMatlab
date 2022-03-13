% Logistic Regression

% The dataset consists of 80 students: 40 who admitted to college and 40 who did not
%
% x = student's score in two exams = (results1, results2)
% y = label of whether the student admitted to college
%


clear all; close all; clc

x = load('log_regressionx.dat'); 
y = load('log_regressiony.dat');

[m, n] = size(x);

% Add intercept term to x
X = [ones(m, 1), x];

% Visualize the datapoints
figure
plot(X(find(y), 2), X(find(y),3), '+')
hold on
plot(X(find(y == 0), 2), X(find(y == 0), 3), 'o')
xlabel('Result 1')
ylabel('Result 2')
legend('Admitted', 'Not admitted')
title({'Training Data'})

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 1. Using 'fitglm' function: "fitglm(x,y)" returns a generalized linear model of 
% the responses y, fit to the data matrix X.
B= fitglm(X,y,'linear', 'distr', 'binomial' )
coeffs=B.Coefficients.Estimate;


% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./coeffs(4)).*(coeffs(3).*plot_x +coeffs(2));
figure
plot(X(find(y), 2), X(find(y),3), '+')
hold on
plot(X(find(y == 0), 2), X(find(y == 0), 3), 'o')
hold on
xlabel('Result 1')
ylabel('Result 2')
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
title({'Using fitglm function'})
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 2. Using Stochastic Gradient Descent

% Scale features and set them to zero mean
mu = mean(x);
sigma = std(x);

x_scaled = (x-mu)./sigma;
X_scaled = [ones(m, 1), x_scaled];

% Save unscaled features to calculate the parameters from the normal equation later
X_unscaled = X;

% Initialize fitting parameters
sgd_theta = zeros(3,1);
grad = zeros(3,1);
iterations = 1500;
alpha = 0.1;

% Define the sigmoid function
g = inline ('1.0 ./(1.0 + exp (-z))'); 

% SGD IMPLEMENTATION HERE

for num_iterations = 1:iterations
   h = g(X_scaled*sgd_theta);
       
   grad = (1/m)*(X_scaled.'* (h-y));
   sgd_theta = sgd_theta - alpha*grad; 
end

% Calculate the parameters from the normal equation
theta_normal = zeros(size(X_scaled(1,:))); 
theta_normal(1) = sgd_theta(1) - ((sgd_theta(2)*mu(1)/sigma(1) + sgd_theta(3)*mu(2)/sigma(2)));
theta_normal(2) = sgd_theta(2)/sigma(1);
theta_normal(3) = sgd_theta(3)/sigma(2);

% Plot SGD method result

% Calculate the decision boundary line
plot_y = (-1./theta_normal(3)).*(theta_normal(2).*plot_x +theta_normal(1));
figure
plot(X(find(y), 2), X(find(y),3), '+')
hold on
plot(X(find(y == 0), 2), X(find(y == 0), 3), 'o')
hold on
xlabel('Result 1')
ylabel('Result 2')
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
title({'Using SGD method'})
hold off

% 3. Using Newton's method

% Initialize fitting parameters
n_theta = zeros(3,1);   
iterations = 1500;
J = zeros(iterations,1);

for num_iters = 1:iterations
    h = g(X*n_theta);
  
    % Calculate gradient and hessian.
    grad = (1/m)*(X.'* (h-y));
    H = X.'* diag(h) * diag(1-h) * X;
    
    % Calculate J (for testing convergence)
    J(num_iters) = (1/m)*sum( (-y).*log(h) - (1-y).*log(1-h) );
    
    n_theta = n_theta-H\grad;
end

% Vizualize Newton's method
% Calculate the decision boundary line
plot_y = (-1./n_theta(3)).*(n_theta(2).*plot_x +n_theta(1));
figure
plot(X(find(y), 2), X(find(y),3), '+')
hold on
plot(X(find(y == 0), 2), X(find(y == 0), 3), 'o')
hold on
xlabel('Result 1')
ylabel('Result 2')
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
title({'Using Newton method'})
hold off