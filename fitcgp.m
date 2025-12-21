function [this, varargout] = fitcgp(X, Y, varargin)
%FITCGP Fit a Gaussian Process Classification (GPC) model.
%   MODEL=FITCGP(TBL,Y) returns a GPC model MODEL for predictors in table
%   TBL and response Y. TBL contains the predictor variables. Y can be any
%   of the following:
%     1. A matric with 2 columns (first column contain number of successes
%        and second column contain number of trials for binomial response)
%     2. A vector (logical, categorical with 2 classes,
%       numeric scores of 2 values, or numeric binary labels)
%     3. A variable name (string/char) naming the response column in TBL.
%
%   MODEL is an object of class `ClassificationGP`. This function follows
%   the fitrgp pattern and delegates fitting to `ClassificationGP.fit`.
%
%   MODEL=FITCGP(X,Y) is an alternative syntax that accepts X as an N-by-P
%   matrix of predictors with one row per observation and one column per
%   predictor. Y is the response vector. Use of a matrix X rather than a
%   table TBL saves both memory and execution time.
%
%   MODEL=FITCGP(X,Y,'PARAM1',val1,'PARAM2',val2,...) specifies optional
%   parameter name/value pairs:
%
%       'KernelFunction'   - A string or a function handle specifying form
%                            of the covariance function of the Gaussian
%                            process. Valid values for 'KernelFunction'
%                            are:
%
%           'squaredexponential'    - Squared exponential kernel (Default).
%           'exponential'           - Exponential kernel.
%           'matern32'              - Matern kernel with parameter 3/2.
%           'matern52'              - Matern kernel with parameter 5/2.
%           'rationalquadratic'     - Rational quadratic kernel.
%           'ardexponential'        - Exponential kernel with a separate
%                                     length scale per predictor.
%           'ardsquaredexponential' - Squared exponential kernel with a
%                                     separate length scale per predictor.
%           'ardmatern32'           - Matern kernel with parameter 3/2
%                                     and a separate length scale per
%                                     predictor.
%           'ardmatern52'           - Matern kernel with parameter 5/2
%                                     and a separate length scale per
%                                     predictor.
%           'ardrationalquadratic'  - Rational quadratic kernel with a
%                                     separate length scale per predictor.
%           KFCN                    - A function handle that can be called
%                                     like this:
%
%                                     KMN = KFCN(XM,XN,THETA)
%
%                                     XM is a M-by-D matrix, XN is a N-by-D
%                                     matrix and KMN is a M-by-N matrix of
%                                     kernel products such that KMN(i,j) is
%                                     the kernel product between XM(i,:)
%                                     and XN(j,:). THETA is the R-by-1
%                                     unconstrained parameter vector for
%                                     KFCN.

%       'DistanceMethod'   - Method used to calculate the euclidean
%                            distances for the kernels. Choices are:
%
% 	            'fast'     - Fast Euclidean distance method. (Default)
%               'accurate' - Accurate Euclidean distance method.
%
%       'KernelParameters' - A vector of initial values for the kernel
%                            parameters. Valid value of 'KernelParameters'
%                            depends on the value of 'KernelFunction' as
%                            follows:
%
%           If 'KernelFunction' is:   'KernelParameters' should be:
%           -----------------------   -----------------------------
%           'squaredexponential'    - A 2-by-1 vector PHI such that:
%           'exponential'             PHI(1) = length scale and
%           'matern32'                PHI(2) = signal standard deviation.
%           'matern52'              - The default value of
%                                     'KernelParameters' is:
%                                     PHI = [mean(std(X));std(Y)/sqrt(2)]
%
%           'rationalquadratic'     - A 3-by-1 vector PHI such that:
%                                     PHI(1) = length scale and
%                                     PHI(2) = rational quadratic exponent
%                                     PHI(3) = signal standard deviation
%                                   - The default value of
%                                     'KernelParameters' is:
%                                     PHI = [mean(std(X));1;std(Y)/sqrt(2)]
%
%           'ardexponential'        - A (D+1)-by-1 vector PHI such that:
%           'ardsquaredexponential'   PHI(i) = predictor i length scale and
%           'ardmatern32'             PHI(D+1) = signal standard deviation.
%           'ardmatern52'           - The default value of
%                                     'KernelParameters' is:
%                                     PHI = [std(X)';std(Y)/sqrt(2)]
%
%           'ardrationalquadratic'  - A (D+2)-by-1 vector PHI such that:
%                                     PHI(i) = predictor i length scale and
%                                     PHI(D+1) = rational quadratic exponent
%                                     PHI(D+2) = signal standard deviation
%                                   - The default value of
%                                     'KernelParameters' is:
%                                     PHI = [std(X)';1;std(Y)/sqrt(2)]
%
%           Function handle KFCN    - A R-by-1 vector as the initial value
%                                     of the unconstrained parameter vector
%                                     THETA parameterizing KFCN.
%                                   - When 'KernelFunction' is a function
%                                     handle, you must supply
%                                     'KernelParameters'.
%
%       'BasisFunction'    - A string or a function handle specifying form
%                            of the explicit basis in the Gaussian process
%                            model. An explicit basis function adds the
%                            term H*BETA to the Gaussian process model
%                            where H is a N-by-P basis matrix and BETA is a
%                            P-by-1 vector of basis coefficients. Valid
%                            values for 'BasisFunction' are:
%
%                   'none'          - H = zeros(N,0).
%                   'constant'      - H = ones(N,1) (Default).
%                   'linear'        - H = [ones(N,1),X].
%                   'purequadratic' - H = [ones(N,1),X,X.^2].
%                   HFCN            - A function handle that can be called
%                                     like this:
%
%                                       H = HFCN(X)
%
%                                     X is a N-by-D matrix of predictors
%                                     and H is a N-by-P matrix of basis
%                                     functions.
%                            If there are categorical predictors, then X in
%                            the above expressions includes dummy variables
%                            for those predictors and D is the number of
%                            predictor columns including the dummy
%                            variables.
%
%       'Beta'             - Initial value for the coefficient vector BETA
%                            for the explicit basis. If the basis matrix H
%                            is N-by-P then BETA must be a P-by-1 vector.
%                            Default is zeros(P,1). The initial value of
%                            'Beta' is used only when 'FitMethod' is 'none'
%                            (see below).
%
%       'Lambda'           - Either 'auto' or a non-negative scalar values
%                            specifying the jitter added to the Kernel
%                            Matrix to force it SPD. Default: 0
%
%       'ConstantLambda'   - A scalar logical specifying whether the
%                            'Lambda' parameter should be held constant
%                            during fitting. Default: true.
%
%       'Inference'        - Method used to approximate distribution using
%                            inference. Choices are:
%
% 	            'logit'  - Laplace approximation with logit link. (Default)
%               'probit' - Laplace approximation with probit link.
% 		        'ep'     - Expectation Propagation (EP). (Binary input only)
%
%       'InferenceOptions' - Struct that contains the options of the
%                            interference methods. Fields are
%
%               'MaxIter' : Maximum number of iterations. Default is 50.
%               'Tol'     : Tolerance. Default is 1e-6.
%               'Damping' : Tolerance. Default is 0.8. (used by EP only)
%
%       'ProbitScaling'   - String specifying the multplier of variance
%                           correction for scores calculates based on the
%                           'logit' inference method. Choices are
%
% 	            'none'    - No variance correction . (Default)
%               'slope'   - Multiplier is SQRT(PI/8) based on matching
%                           logit and probit derivatives functions.
% 		        'minimax' - Multiplier is 0.608 based on minimizing the max
%                           difference between logit and probit functions.
%
%       'FitMethod'        - Method used to estimate parameters of the
%                            Gaussian process model. Choices are:
%
%               'none'  - No estimation (uses initial parameter values).
%               'exact' - Exact Gaussian Process Classification.
%               'sd'    - Subset of Datapoints approximation.
%
%                            Default is 'exact' for N <= MIN(2000,
%                            ActiveSetSize) and 'sd' otherwise.
%
%       'PredictMethod'    - Method used to make predictions from a
%                            Gaussian process model given the parameters.
%                            Choices are:
%
%               'exact' - Exact Gaussian Process Classification.
%               'sd'    - Subset of Datapoints approximation.
%
%                            Default is 'exact' for N <= MIN(2000,
%                            ActiveSetSize) and 'sd' otherwise.
%
%       'ActiveSet'        - A vector of integers of length M where
%                            1 <= M <= N indicating the observations that
%                            are in the active set. 'ActiveSet' should not
%                            have duplicate elements and its elements must
%                            be integers from 1 to N. Alternatively,
%                            'ActiveSet' can also be a logical vector of
%                            length N with at least 1 true element. If you
%                            supply 'ActiveSet' then 'ActiveSetSize' and
%                            'ActiveSetMethod' has no effect. Default is [].
%
%       'ActiveSetSize'    - An integer M with 1 <= M <= N specifying the
%                            size of the active set for sparse fit methods
%                            like 'sd', 'sr' and 'fic'. Typical values of
%                            'ActiveSetSize' are a few hundred to a few
%                            thousand. Default is min(2000,N).
%
%       'ActiveSetMethod'  - A string specifying the active set selection
%                            method. Choices are:
%
%               'first'      - Select first M data points.
%               'last'       - Select last M data points.
%               'qr'         - Select M most dominant points based on QR.
%               'random'     - Random selection of M data points.
%
%                            All active set selection methods (except
%                            'random') require the storage of a N-by-M
%                            matrix where M is the size of the active set
%                            and N is the number of observations. Default
%                            is 'random'.
%
%       'Standardize'      - Logical scalar. If true, standardize X by
%                            centering and dividing columns by their
%                            standard deviations. Default is false.
%
%       'Verbose'          - Verbosity level, one of: 0 or 1.
%                            o If 'Verbose' is > 0, iterative diagnostic
%                              messages related to parameter estimation are
%                              displayed on screen.
%                            o If 'Verbose' is 0, diagnostic messages
%                              related parameter estimation are displayed
%                              depending on the value of 'Display' in
%                              'OptimizerOptions'. Default is 0.
%
%       'Weights'          - Vector of observation weights, one weight per
%                            observation. Default: ones(size(X,1),1).
%
%       'Optimizer'        - A string specifying the optimizer to use for
%                            parameter estimation. Choices include
%                            'fminsearch', 'quasinewton', 'lbfgs',
%                            'fminunc' and 'fmincon'. Use of 'fminunc' and
%                            'fmincon' requires an Optimization Toolbox
%                            license. Default is 'quasinewton'.
%
%       'OptimizerOptions' - A structure or object containing options for
%                            the chosen optimizer. If 'Optimizer' is
%                            'fminsearch', 'OptimizerOptions' must be
%                            created using OPTIMSET. If 'Optimizer' is
%                            'quasinewton' or 'lbfgs', 'OptimizerOptions'
%                            must be created using STATSET('fitcgp'). If
%                            'Optimizer' is 'fminunc' or 'fmincon',
%                            'OptimizerOptions' must be created using
%                            OPTIMOPTIONS. Default depends on the chosen
%                            value of 'Optimizer'.
%
%       'PredictorNames'   - A string array or cell array of names for the
%                            predictor variables, in the order in which they appear
%                            in X. Default: {'x1','x2',...}. For a table
%                            TBL, these names must be a subset of the
%                            variable names in TBL, and only the selected
%                            variables are used. Not allowed when Y is a
%                            formula. Default: all variables other than Y.
%
%       'ResponseName'     - Name of the response variable Y, a string. Not
%                            allowed when Y is a name or formula.
%                            Default: 'Y'
%
%       'OptimizeHyperparameters'
%                      - Hyperparameters to optimize. Either 'none',
%                        'auto', 'all', a string/cell array of eligible
%                        hyperparameter names, or a vector of
%                        optimizableVariable objects, such as that returned
%                        by the 'hyperparameters' function. To control
%                        other aspects of the optimization, use the
%                        HyperparameterOptimizationOptions name-value pair.
%                        'auto' is equivalent to {'KernelScale',
%                        'Standardize'}. 'all' is equivalent to
%                        {'BasisFunction', 'KernelFunction', 'KernelScale',
%                        'Standardize'}. Note: When 'KernelScale' is
%                        optimized, the 'KernelParameters' argument to
%                        fitrgp is used to specify the value of the kernel
%                        scale parameter, which is held constant during
%                        fitting. In this case, all input dimensions are
%                        constrained to have the same KernelScale value.
%                        KernelScale cannot be optimized for any of the ARD
%                        kernels. Default: 'none'.
%
%  [MODEL, HYPEROPTR] = FITCGP(...) returns bayesopt results (struct) when
%  'OptimizeHyperparameters' is not 'none'. Otherwise HYPEROPTR is [].
%
% Key name-value pairs (subset):
%  - 'KernelFunction', 'KernelParameters', 'DistanceMethod'
%  - 'BasisFunction', 'Beta'
%  - 'Sigma', 'ConstantSigma'
%  - 'Inference': 'Logit' (Default) | 'Probit' | 'EP'
%  - 'InferenceOptions': struct with fields like MaxIter, Tol, Damping
%  - 'ProbitScaling' : 'None' (Default) | 'Slope' | 'Minimax'
%  - 'ActiveSetSize', 'ActiveSetMethod'
%  - 'Optimizer', 'OptimizerOptions'
%  - 'OptimizeHyperparameters', 'HyperparameterOptimizationOptions'
%
%
%   Example: Train a GPC model on example data.
%       % Basic binary classification
%       load fisheriris
%       X = meas(1:100, :);
%       y = categorical(species(1:100));
%       gpc = fitcgp(X, y, 'Inference', 'Probit');
%       [labels, scores] = predict(gpc, X);
%
%       % With hyperparameter optimization
%       gpc = fitcgp(X, y, ...
%                   'OptimizeHyperparameters', 'auto', ...
%                   'HyperparameterOptimizationOptions', ...
%                   struct('MaxObjectiveEvaluations', 30));
%
%       % Using different kernels
%       gpc_se = fitcgp(X, y, 'KernelFunction', 'squaredexponential');
%       gpc_m32 = fitcgp(X, y, 'KernelFunction', 'matern32');
%       gpc_ard = fitcgp(X, y, 'KernelFunction', 'ardsquaredexponential');
%
%       % Custom basis function
%       customBasis = @(X) [ones(size(X,1),1), X, X.^2];
%       gpc = fitcgp(X, y, 'BasisFunction', customBasis);
%
% See also: ClassificationGP

%
%       'ConstantKernelParameters'
%                          - A logical vector indicating which kernel
%                            parameters should be held constant during
%                            fitting. See the 'KernelParameters' parameter
%                            for the required dimensions of this argument.
%                            Default: false for all kernel parameters.


[this, hyperOptResults] = ClassificationGP.fit(X, Y, varargin{:});

if nargout > 1
    varargout{1} = hyperOptResults;
end
end