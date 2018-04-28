% Initialization
clear; close all; clc;

% Set up parameters
input_layer_size = 784;  % 28x28 input images
num_labels = 10;

% Load training data
X = loadMNISTImages('train-images.idx3-ubyte');
y = loadMNISTLabels('train-labels.idx1-ubyte');
m = size(X, 1);
n = size(X, 2);

% Split training data into primitive training set & validation set
Xtrain = X(1 : (end - 9999), :);
ytrain = y(1 : (end - 9999), :);
Xval = X((end - 10000) :  end, :);
yval = y((end - 10000) : end);

% Training with one-vs-all method
lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
Thetas = zeros(num_labels, n + 1, length(lambdas));
for iter = 1 : length(lambdas),
	Thetas(:, :, iter) = oneVsAll(Xtrain, ytrain, num_labels, lambdas(iter));
end;

fprintf('Program paused. Press enter to continue.\n');
pause;

% Report training set accuracy
for iter = 1 : length(lambdas),
	predTrain = predictOneVsAll(Thetas(:, :, iter), Xtrain);
	fprintf('\nTraining set accuracy for lambda = %f is: %f\n', ...
		lambdas(iter), mean(double(predTrain == ytrain)) * 100);
end;

% Save the parameters learned
save lrParams.mat Thetas;
