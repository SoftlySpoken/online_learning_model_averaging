% Initialization
clear; close all; clc;

% Set up parameters
input_layer_size = 784;  % 28x28 input images
hidden_layer_size = 50;  % 50 hidden units (single layer)
num_labels = 10;  % 10 output units

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

% Randomly initialize neural network parameters
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];  % "Unroll" parameters

% Train neural network
fprintf('\nTraining neural network:\n')
options = optimset('MaxIter', 150);
lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
Theta1s = zeros(hidden_layer_size, input_layer_size + 1, length(lambdas));
Theta2s = zeros(num_labels, hidden_layer_size + 1, length(lambdas));
for iter = 1 : length(lambdas),
	costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
	num_labels, Xtrain, ytrain, lambdas(iter));
	[nn_params, cost] = fmincg(costFunc, initial_nn_params, options);
	Theta1s(:, :, iter) = reshape(nn_params(1 : (hidden_layer_size * ...
		(input_layer_size + 1))), hidden_layer_size, (input_layer_size + 1));
	Theta2s(:, :, iter) = reshape(nn_params((hidden_layer_size * ...
		(input_layer_size + 1) + 1) : end), num_labels, (hidden_layer_size + 1));
end;

fprintf('Program paused. Press enter to continue.\n');
pause;

% Report Training set accuracy
for iter = 1 : length(lambdas),
	predTrain = predict(Theta1s(:, :, iter), Theta2s(:, :, iter), Xtrain);
	fprintf('\nTraining set accuracy for lambda = %f is: %f\n', ...
		lambdas(iter), mean(double(predTrain == ytrain)) * 100);
end;
% Save regularization parameters
save regParams.mat lambdas;

% Save the parameters learned
save nnParams.mat Theta1s Theta2s;

% Save validation data
save validationSet.mat Xval yval;




