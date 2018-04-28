function p = predictOneVsAllVec(all_theta, x)

num_labels = size(all_theta, 1);
p = zeros(1, num_labels);
p = sigmoid([1, x] * all_theta');

end