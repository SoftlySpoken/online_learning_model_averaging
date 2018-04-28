# Model Averaging with Online Learning Algorithms
A series of experiments conducted to compare the performance of a traditional model selection method and model averaging with online learning algorithms applied to a set of machine learning algorithms, namely, logistic regression and neural networks with different parameter settings, on the classification problem of the MNIST database of handwritten digits.

Coursework for "Online Learning and Optimization" taught by Dr. Andras Gyorgy from Imperial College London during Peking University's 2017 Summer Session. Uploading for archival purposes.

Programs in this folder are all written in GNU Octave, version 3.8.0. It is highly probable, but not guaranteed, that they will run correctly on later versions of GNU Octave and versions of Matlab currently in distribution.

trainLogisticRegression.m and trainNeuralNetwork.m, if run directly, will train 10 logistic regression models and 10 neural networks respectively when run under this directory, the parameters of which are specified in the report. Note that because of the random initialization of weights in trainNeuralNetwork.m, the training result may differ slightly with every run.

These two scripts will eventually save lrParams.mat, nnParams.mat, regParams.mat and validationSet.mat to the current directory. These files that have produced the results in my report are readily provided here.

validation.m, if run directly, will load the four aforementioned .mat files, evaluate all 20 models on the validation set, select the best expert, and evaluate it on the test set.

WMA1.m, WMA1Norm.m, WMA2.m, WMA2Norm.m, EWA.m and REWA.m are six online learning algorithms experimented on, coded as functions receiving a single argument eta. They can be run separately, performing model averaging with different algorithms given eta. Or findBestEta.m, a function receiving a function handle and the number of iterations as arguments, can help find the value of eta that yields the highest test accuracy (and also output test accuracies of each eta experimented on) for one particular algorithm.

Other scripts are functions called in the aforementioned scripts and need not be run separately.

References:

fmincg.m is by Rasmussen, C.E., 2002. Detailed copyright message can be found in the script.

loadMNISTImages.m and loadMNISTLabels.m are mnistHelper functions offered on http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset , used for loading the MNIST datasets.

lrCostFunction.m, nnCostFunction.m, oneVsAll.m, predict.m, predictOneVsAll.m, randInitializeWeights.m, sigmoid.m and sigmoidGradient.m are part of the programming assignments of the Machine Learning course taught by Andrew Ng on Coursera, meaning that some parts of them are starter code and others are my work. Comments in these scripts should be sufficient for determining which parts are not my original work.

For the course report that demonstrates my findings, please see `report.pdf`.
