
clear; close all; clc

%% Load Data
data = load('LeukemiaTraining.txt');
X = data(:, 2:101); y = data(:, 1);
data_test = load('LeukemiaTesting.txt');
X_test = data_test(:, 2:101);  y_test = data_test(:, 1);


s = size(y,1);
y1 = zeros(1,1);
for i = 1:s
    if(y(i)==0)
        y1 = [y1;1];
    else
        y1 = [y1;0];
    end
end
y1 = y1(2:s+1);

y2 = zeros(1,1);
for i = 1:s
    if(y(i)==1)
        y2 = [y2;1];
    else
        y2 = [y2;0];
    end
end
y2 = y2(2:s+1);
y3 = zeros(1,1);
for i = 1:s
    if(y(i)==2)
        y3 = [y3;1];
    else
        y3 = [y3;0];
    end
end
y3 = y3(2:s+1);
        

%% ============ Compute Cost and Gradient ============

[m, n] = size(X);
[m1, n1] = size(X_test);
X = [ones(m, 1) X];
X_test = [ones(m1, 1) X_test];
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost1, grad1] = costFunction(initial_theta, X, y1);
[cost2, grad2] = costFunction(initial_theta, X, y2);
[cost3, grad3] = costFunction(initial_theta, X, y3);

 fprintf('Cost at initial theta(zeros) for class 0: %f\n', cost1);
 fprintf('Cost at initial theta(zeros) for class 1: %f\n', cost2);
 fprintf('Cost at initial theta(zeros) for class 2: %f\n', cost3);
% fprintf('Gradient at initial theta (zeros): \n');
% fprintf(' %f \n', grad);



%% ============= Optimizing using fminunc or fminsearch =============
 options = optimset('GradObj', 'on', 'MaxIter', 400, 'TolFun',1e-6, 'TolX',1e-7);

[theta1, cost1] = ...
	fminsearch(@(t)(costFunction(t, X, y1)), initial_theta, options);
[theta2, cost2] = ...
	fminsearch(@(t)(costFunction(t, X, y2)), initial_theta, options);
[theta3, cost3] = ...
	fminsearch(@(t)(costFunction(t, X, y3)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc for class 0: %f\n', cost1);
fprintf('Cost at theta found by fminunc for class 1: %f\n', cost2);
fprintf('Cost at theta found by fminunc for class 2: %f\n', cost3);
% fprintf('theta: \n');
% fprintf(' %f \n', theta);



%% ============== Predict and Accuracies ==============

p = predict(theta1,theta2,theta3, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
ptest = predict(theta1,theta2,theta3, X_test);
fprintf('Testing Accuracy: %f\n', mean(double(ptest == y_test)) * 100);

