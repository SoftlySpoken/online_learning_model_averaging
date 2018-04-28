% Initialization
clear; close all; clc;

% Set up parameters
input_layer_size = 784;  % 28x28 input images
hidden_layer_size = 50;  % neural network has 50 hidden units (single layer)
num_labels = 10;

% Load regularization parameters
load regParams.mat;  % lambdas

% Load learned parameters
load lrParams.mat;  % Thetas
load nnParams.mat;  % Theta1s, Theta2s

% Load training data
X = loadMNISTImages('train-images.idx3-ubyte');
y = loadMNISTLabels('train-labels.idx1-ubyte');
m = size(X, 1);
n = size(X, 2);

% Load validation data
load validationSet.mat;  % Xval, yval

% Load test data
Xtest = loadMNISTImages('t10k-images.idx3-ubyte');
ytest = loadMNISTLabels('t10k-labels.idx1-ubyte');

% Validation
valAcc = zeros((2 * length(lambdas)), 1);  % validation accuracy of each expert
fprintf('Validating Logistic Regression...\n');
for iter = 1 : length(lambdas),
	predVal = predictOneVsAll(Thetas(:, :, iter), Xval);
	valAcc(iter) = mean(double(predVal == yval)) * 100;
	fprintf('Validation set accuracy for lambda = %f is: %f\n', ...
		lambdas(iter), valAcc(iter));
end;
fprintf('Validating Neural Networks...\n');
for iter = 1 : length(lambdas),
	predVal = predict(Theta1s(:, :, iter), Theta2s(:, :, iter), Xval);
	valAcc(iter + length(lambdas)) = mean(double(predVal == yval)) * 100;
	fprintf('Validation set accuracy for lambda = %f is: %f\n', ...
		lambdas(iter), valAcc(iter + length(lambdas)));
end;

% Select, retrain and test the best expert on full training set
[bestValAcc, bestIdx] = max(valAcc);
if bestIdx <= length(lambdas),  % the best expert is a logistic regression model
	% Retrain
	theta = oneVsAll(X, y, num_labels, lambdas(bestIdx));
	% Test
	predTest1 = predictOneVsAll(theta, Xtest);
	testAcc1 = mean(double(predTest1 == ytest)) * 100;
	fprintf('The best expert is a logistic regression model, lambda = %f\n', ...
		lambdas(bestIdx));
	fprintf('Its test set accuracy is: %f\n', testAcc1);
else  % the best expert is a neural network
	% Randomly initialize neural network parameters
	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
	initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];  % "Unroll" parameters
	% Retrain
	options = optimset('MaxIter', 150);
	costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
	num_labels, X, y, lambdas(bestIdx - length(lambdas)));
	[nn_params, cost] = fmincg(costFunc, initial_nn_params, options);
	theta1 = reshape(nn_params(1 : (hidden_layer_size * ...
		(input_layer_size + 1))), hidden_layer_size, (input_layer_size + 1));
	theta2 = reshape(nn_params((hidden_layer_size * ...
		(input_layer_size + 1) + 1) : end), num_labels, (hidden_layer_size + 1));
	% Test
	predTest1 = predict(theta1, theta2, Xtest);
	testAcc1 = mean(double(predTest1 == ytest)) * 100;
	fprintf('The best expert is a neural network, lambda = %f\n', ...
		lambdas(bestIdx - length(lambdas)));
	fprintf('Its test set accuracy is: %f\n', testAcc1);
end;




