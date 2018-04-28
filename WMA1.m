% Turn off automatic broadcast warning
warning("off", "Octave:broadcast");

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

% Load test data
Xtest = loadMNISTImages('t10k-images.idx3-ubyte');
ytest = loadMNISTLabels('t10k-labels.idx1-ubyte');

% WMA, weights not normalized, zero-one loss update
eta = 10;  % experiment with {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}. beta = exp(-eta)
% Initialize weights to 1
weights = zeros((2 * length(lambdas)), 1);
weights(1 : (2 * length(lambdas))) = 1;
expertPred = zeros((2 * length(lambdas)), num_labels);  % the expert predictions matrix
expertFinalPred = zeros((2 * length(lambdas)), 1);  % the expert predictions vector (single labels)
expertLoss = zeros((2 * length(lambdas)), 1);  % the expert loss vector (zero-one loss)
labelsWeight = zeros(num_labels, 1);  % the accumulated weight of each label
pred = 0;  % the combined prediction
testAcc = 0;  % test accuracy
for t = 1 : size(Xtest, 1),  % loop over time
	labelsWeight(1 : num_labels) = 0;  % initialize labelsWeight
	for iter = 1 : (2 * length(lambdas)),  % loop over experts
		if iter <= length(lambdas),  % the expert is a logistic regression model
			expertPred(iter, :) = predictOneVsAllVec(Thetas(:, :, iter), Xtest(t, :));
		else  % the expert is a neural network
			expertPred(iter, :) = predictVec(Theta1s(:, :, (iter - length(lambdas))), ...
				Theta2s(:, :, (iter - length(lambdas))), Xtest(t, :));
		end;
	end;
	expertFinalPred = nthargout(2, @max, expertPred, [], 2);
	% Accumulate weights for each label
	for iter = 1 : (2 * length(lambdas)),
		labelsWeight(expertFinalPred(iter)) += weights(iter);
	end;
	% Predict the label with maximum weight
	pred = nthargout(2, @max, labelsWeight);
	% Output for debugging
	%fprintf('\nt = %d\n', t);
	%fprintf('\nExperts predictions are:\n');
	%disp(expertFinalPred);
	%fprintf('\nWeights of experts are:\n');
	%disp(weights);
	%fprintf('\nWeights of labels are:\n');
	%disp(labelsWeight);
	%fprintf('\nCombined prediction is:\n');
	%disp(pred);
	%fprintf('\nTrue label is:\n');
	%disp(ytest(t));
	testAcc += (pred == ytest(t));  % accumulate test accuracy
	% Derive binary representation of ytest(t)
	yt = zeros(1, num_labels);
	yt(ytest(t)) = 1;
	% Calculate loss of each expert
	expertLoss = (ytest(t) ~= expertFinalPred);
	% Update weights
	weights = weights .* exp(-eta * expertLoss);
	%fprintf('\nProgram paused. Press enter to go to the next timestep.\n');
	%pause;
end;

% Report final test accuracy
testAcc = testAcc / size(Xtest, 1) * 100;
fprintf('WMA1');
fprintf('\neta = %f', eta);
fprintf('\nCombined test accuracy is: %f\n', testAcc);



