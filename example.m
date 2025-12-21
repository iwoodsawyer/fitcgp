% Basic binary classification
load fisheriris
X = meas(1:100, :);
y = categorical(species(1:100));
gpc = fitcgp(X, y, 'Inference', 'Probit');
[labels, scores] = predict(gpc, X);

% With hyperparameter optimization
gpc = fitcgp(X, y, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', ...
    struct('MaxObjectiveEvaluations', 30));

% Using different kernels
gpc_se = fitcgp(X, y, 'KernelFunction', 'squaredexponential');
gpc_m32 = fitcgp(X, y, 'KernelFunction', 'matern32');
gpc_ard = fitcgp(X, y, 'KernelFunction', 'ardsquaredexponential');

% Custom basis function
customBasis = @(X) [ones(size(X,1),1), X, X.^2];
gpc = fitcgp(X, y, 'BasisFunction', customBasis);