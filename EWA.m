function testAcc = EWA(eta)

% Turn off automatic broadcast warning
warning("off", "Octave:broadcast");

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

% EWA
% Initialize normalized weights to 1 / N (N = 2 * length(lambdas))
normWeights = zeros((2 * length(lambdas)), 1);
normWeights(1 : (2 * length(lambdas))) = 1 / (2 * length(lambdas));
expertPred = zeros((2 * length(lambdas)), num_labels);  % the expert predictions matrix
expertLoss = zeros((2 * length(lambdas)), 1);  % the expert loss vector
predVec = zeros(1, num_labels);  % the combined prediction vector
testAcc = 0;  % test accuracy
for t = 1 : size(Xtest, 1),  % loop over time
	predVec(1 : num_labels) = 0;  % initialize prediction vector
	for iter = 1 : (2 * length(lambdas)),  % loop over experts
		if iter <= length(lambdas),  % the expert is a logistic regression model
			expertPred(iter, :) = predictOneVsAllVec(Thetas(:, :, iter), Xtest(t, :));
		else  % the expert is a neural network
			expertPred(iter, :) = predictVec(Theta1s(:, :, (iter - length(lambdas))), ...
				Theta2s(:, :, (iter - length(lambdas))), Xtest(t, :));
		end;
	end;
	predVec = normWeights' * expertPred;  % make combined prediction
	testAcc += ((nthargout(2, @max, predVec)) == ytest(t));  % accumulate test accuracy
	% Derive binary representation of ytest(t)
	yt = zeros(1, num_labels);
	yt(ytest(t)) = 1;
	% Calculate loss of each expert
	expertLoss = sum((-yt .* log(expertPred) - (1 - yt) .* log(1 - expertPred)), 2);
	% Update weights
	normWeights = normWeights .* exp(-eta * expertLoss);
	normWeights = normWeights / sum(normWeights);
end;

% Report final test accuracy
testAcc = testAcc / size(Xtest, 1) * 100;
fprintf('EWA');
fprintf('\neta = %f', eta);
fprintf('\nCombined test accuracy is: %f\n', testAcc);

end;


