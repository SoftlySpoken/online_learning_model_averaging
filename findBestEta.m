function findBestEta(algo, numIterations)

%% findBestEta(algo, numIterations) finds the best
%% value of the learning rate eta (i.e. which yields
%% the highest test accuracy) in the interval (0, 10]
%% given the number of iterations and the algorithm
%% tested.

unit = 10 / numIterations;
bestEta = 0;  % the best value of eta
bestAcc = 0;  % the highest test accuracy
for iter = 1 : numIterations,
	acc = algo(unit * iter);
	if acc > bestAcc,
		bestAcc = acc;
		bestEta = unit * iter;
	end;
end;

fprintf('\nThe best eta for '); disp(algo);
fprintf('is %f, yielding test accuracy %f\n', bestEta, bestAcc);

end;