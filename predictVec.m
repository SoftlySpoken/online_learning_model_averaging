function p = predictVec(Theta1, Theta2, x)

num_labels = size(Theta2, 1);
p = zeros(1, num_labels);
h1 = sigmoid([1, x] * Theta1');
h2 = sigmoid([1, h1] * Theta2');
p = h2;

end