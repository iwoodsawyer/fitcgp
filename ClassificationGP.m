classdef ClassificationGP
%CLASSIFICATIONGP Gaussian Process Classification (GPC) model (binary).
%
%   ClassificationGP is a Gaussian process model for classification. This 
%   model can predict response given new data. This model also stores data 
%   used for training and can compute resubstitution predictions.
%
%   An object of this class cannot be created by calling the constructor.
%   Use FICRGP to create a ClassificationGP object by fitting a GPC model
%   totraining data.
%
%   This class provides a fitrgp-style API for **binary** Gaussian process
%   classification. It is intended to resemble MATLAB's `RegressionGP` and
%   `fitrgp` interface and conventions, but implements classification via
%   approximate inference using Laplace or Expectation Propagation (EP):
%
%   ClassificationSVM properties:
%       NumObservations       - Number of observations.
%       X                     - Matrix of predictors used to train this model.
%       Y                     - Observed response used to train this model.
%       W                     - Weights of observations used to train this model.
%       ModelParameters       - GPC parameters.
%       PredictorNames        - Names of predictors used for this model.
%       ExpandedPredictorNames - Names of expanded predictors.
%       ResponseName          - Name of the response variable.
%       ResponseTransform     - Transformation applied to predicted regression response.
%       KernelFunction        - Kernel function used in this model.
%       KernelInformation     - Information about parameters of this kernel function.
%       BasisFunction         - Basis function used in this model.
%       Beta                  - Estimated value of basis function coefficients.
%       Lambda                - Jitter to force positive definete of kernel matrix.
%       PredictorLocation     - A vector of predictor means (if standardization is used).
%       PredictorScale        - A vector of predictor standard deviations (if standardization is used).
%       Alpha                 - Vector of weights for computing predictions.
%       ActiveSetVectors      - Subset of the training data needed to make predictions.
%       FitMethod             - Method used to estimate parameters.
%       PredictMethod         - Method used to make predictions.
%       ActiveSetMethod       - Method used to select the active set.
%       ActiveSetSize         - Size of the active set.
%       IsActiveSetVector     - Logical vector marking the active set.
%       LogLikelihood         - Maximized marginal log likelihood of the model.
%       ActiveSetHistory      - History of active set selection for sparse methods.
%       RowsUsed              - Logical index for rows used in fit. 
%
%   ClassificationGP methods:
%       compact               - Compact this model.
%       loss                  - Classification loss.
%       predict               - Predicted response of this model.
%       resubLoss             - Resubstitution classification loss.
%       resubPredict          - Resubstitution predictions.
%
% ALGORITHMS:
%   Laplace Approximation:
%     Rasmussen & Williams (2006), "Gaussian Processes for Machine Learning"
%     Section 3.4: Laplace Approximation
%
%   Expectation Propagation:
%     Minka (2001), "Expectation Propagation for approximate Bayesian inference"
%     Rasmussen & Williams (2006), Section 3.6
%
%   Kernel Functions:
%     Standard stationary kernels including squared exponential (RBF),
%     Matérn family (3/2, 5/2), and rational quadratic.
%
% REFERENCES:
%   [1] Rasmussen, C. E., & Williams, C. K. (2006). Gaussian processes 
%       for machine learning. MIT press.
%   [2] Minka, T. P. (2001). Expectation propagation for approximate 
%       Bayesian inference. UAI.
%
% See also: fitcgp

properties (Constant, Access=private)
    DEFAULT_ACTIVE_SET_SIZE = 2000
    MIN_JITTER = 1e-10
    MAX_JITTER = 1e-6
    DEFAULT_MAX_ITER = 50
    DEFAULT_TOL = 1e-6
    DEFAULT_DAMPING = 0.8
    PROBIT_MINIMAX = 0.6080;
    PROBIT_SLOPE = sqrt(pi/8);
end

properties
    % ---- Data / bookkeeping (fitrgp-like public properties) ----
    NumObservations
    X
    Y
    W
    ModelParameters
    PredictorNames
    ExpandedPredictorNames
    ResponseName
    ResponseTransform

    % ---- Model definition (fitrgp-like public properties) ----
    KernelFunction
    KernelInformation
    BasisFunction
    Beta
    Lambda

    % ---- Standardization stats (fitrgp-like public properties) ----
    PredictorLocation
    PredictorScale

    % ---- Posterior representation hooks (fitrgp-like public properties) ----
    Alpha
    ActiveSetVectors

    % ---- Training/inference options and results (fitrgp-like public properties) ----
    Inference                 % 'Laplace' or 'EP' (internal name), derived from user's 'Inference'
    FitMethod
    PredictMethod
    ActiveSetSize
    ActiveSetMethod
    IsActiveSetVector
    LogLikelihood
    ActiveSetHistory
    RowsUsed
    HyperparameterOptimizationResults
end

properties (Access=private)
    % ---- Internal caches not meant to be user-facing ----
    Posterior_                % Struct holding inference-specific posterior quantities
    Classes_                  % categorical array of length 2 defining class order
    Link_                     % 'logit' or 'probit'
    InferenceOptions_         % tolerance, max iterations, damping, etc.
    Standardize_ logical
    DistanceMethod_ char
    ProbitScaling_ char
end

methods (Static)
    function [this, hyperOptResults] = fit(X, Y, varargin)
        %FIT Fit a Gaussian Process Classification model.
        %
        % This static constructor-style method matches `RegressionGP.fit`
        % conventions (called by `fitcgp`).

        ClassificationGP.validateInputs(X, Y);
        args = ClassificationGP.parseFitArgs(X, Y, varargin{:});

        % Convert table inputs, extract predictor names, normalize response, etc.
        [Xraw, y01, w, meta] = ClassificationGP.prepareData(args);

        % Create the model object and fill public, fitrgp-like properties.
        this = ClassificationGP();
        this.NumObservations = size(Xraw,1);
        this.X = Xraw;
        this.Y = meta.OriginalY;
        this.W = w;
        this.PredictorNames = meta.PredictorNames;
        this.ExpandedPredictorNames = meta.PredictorNames;
        this.ResponseName = meta.ResponseName;
        this.ResponseTransform = args.ResponseTransform;
        this.RowsUsed = true(size(Xraw,1),1);

        this.KernelFunction = args.KernelFunction;
        this.KernelInformation = struct('Name',args.KernelFunction,...
            'KernelParameters',args.KernelParameters);
        this.BasisFunction = args.BasisFunction;

        % Lambda is used as diagonal jitter (numeric stability).
        this.Lambda = args.Lambda;

        % Derived inference selections:
        %   args.InferenceMethod is 'Laplace' or 'EP'
        %   args.Likelihood is 'logit' or 'probit'
        this.Inference = args.InferenceMethod;
        this.Link_ = args.Likelihood;

        this.FitMethod = args.FitMethod;
        this.PredictMethod = args.PredictMethod;
        this.ActiveSetSize = args.ActiveSetSize;
        this.ActiveSetMethod = args.ActiveSetMethod;

        this.InferenceOptions_ = args.InferenceOptions;
        this.Standardize_ = args.Standardize;
        this.DistanceMethod_ = char(string(args.DistanceMethod));
        this.ProbitScaling_ = char(string(args.ProbitScaling));
        this.Classes_ = meta.Classes;

        hyperOptResults = [];

        % ---- Bayesopt Hyperparameter Optimization ----
        % ---- Optional: bayesopt outer loop ----
        % If enabled, bayesopt will select some hyperparameters (e.g., Lambda,
        % KernelFunction, Standardize...) and then we refit at the optimum.
        if ~strcmpi(string(args.OptimizeHyperparameters), "none")
            [bestArgs, hyperOptResults] = ClassificationGP.runBayesopt(Xraw, y01, w, meta, args);
            args = bestArgs;

            % Update key model-visible selections to the optimized choices.
            this.HyperparameterOptimizationResults = hyperOptResults;
            this.KernelFunction = args.KernelFunction;
            this.BasisFunction = args.BasisFunction;
            this.Standardize_ = args.Standardize;
            this.Inference = args.InferenceMethod;
            this.Link_ = args.Likelihood;
        end

        % ---- Standardization ----
        % Keep this.X raw for user parity; do math on standardized copy.
        if this.Standardize_
            [Xstd, mu, sig] = ClassificationGP.standardizeX(Xraw);
            this.PredictorLocation = mu;
            this.PredictorScale = sig;
        else
            Xstd = Xraw;
            this.PredictorLocation = zeros(1,size(Xraw,2));
            this.PredictorScale = ones(1,size(Xraw,2));
        end

        % ---- Active set selection ----
        % This provides a compute-limiting subset of training points used for
        % posterior inference and prediction (similar spirit to sparse methods).
        [XaStd, ya01, wa, isActive] = ClassificationGP.selectActiveSet(Xstd, y01, w, args);
        this.IsActiveSetVector = isActive;
        this.ActiveSetVectors = Xstd;

        % ---- Basis function setup ----
        % Explicit basis mean: m(x)=H(x)*Beta. This matches fitrgp style.
        Ha = ClassificationGP.basisMatrix(XaStd, args.BasisFunction);
        beta0 = ClassificationGP.ensureBetaSize(args.Beta, size(Ha,2));
        this.Beta = beta0;

        % ---- FitMethod='none' => no hyperparameter estimation ----
        % We only run inference with the provided initial parameters.
        if strcmpi(string(args.FitMethod), "none")
            [post, ll] = ClassificationGP.runInferenceFromArgs(args, XaStd, ya01, wa, Ha, beta0);
            this.Posterior_ = post;
            this.LogLikelihood = ll;
            this.Alpha = post.alpha;
            this.ModelParameters = ClassificationGP.buildModelParameters(args, this);
            return;
        end

        % ---- Local evidence maximization ----
        % Optimize kernel parameters / Beta / Lambda by maximizing the approximate
        % marginal log likelihood (Laplace/EP evidence approximation).
        args = ClassificationGP.optimizeLocally(args, XaStd, ya01, wa, Ha);

        % Copy optimized parameters back to public properties.
        this.Lambda = args.Lambda;
        this.KernelFunction = args.KernelFunction;
        this.BasisFunction = args.BasisFunction;
        this.Beta = args.Beta(:);
        this.KernelInformation.KernelParameters = args.KernelParameters;

        % ---- Final inference with optimized parameters ----
        [post, ll] = ClassificationGP.runInferenceFromArgs(args, XaStd, ya01, wa, Ha, this.Beta);
        this.Posterior_ = post;
        this.LogLikelihood = ll;
        this.Alpha = post.alpha;

        this.ModelParameters = ClassificationGP.buildModelParameters(args, this);
    end
end

methods
    function compacted = compact(this)
        %COMPACT Create a compact version of this model.
        %
        % Similar in spirit to `compact(RegressionGP)`:
        % removes training data arrays to reduce memory footprint while
        % keeping enough information to predict.
        compacted = this;
        compacted.X = [];
        compacted.Y = [];
        compacted.W = [];
        compacted.NumObservations = 0;
        compacted.RowsUsed = [];
        compacted.HyperparameterOptimizationResults = [];
    end

    function [label, score, ci] = predict(this, Xnew, varargin)
        %PREDICT Predict class label and posterior probability for the positive class.
        %
        %   [LABEL,SCORE] = predict(MODEL,XNEW)
        %   returns:
        %     LABEL : categorical labels (2 classes)
        %     SCORE : N-by-2 probabilities for each class
        %
        %   [LABEL,SCORE,CI] also returns a crude confidence interval for
        %   the positive-class probability (based on approximate latent variance).
        %
        % Name-value:
        %   'Alpha' : significance level in (0,1). Default 0.05 -> 95% CI.

        if (size(Xnew,2) ~= size(this.ActiveSetVectors,2))
            error('ClassificationGP:SizeMismatch', ...
                'Number of columns in Xnew (%d) must equal columns of X (%d).', ...
                size(Xnew,2), size(this.ActiveSetVectors,2));
        end

        ip = inputParser;
        ip.addParameter('Alpha', 0.05, @(a) isnumeric(a) && isscalar(a) && a>0 && a<1);
        ip.parse(varargin{:});
        alphaLevel = ip.Results.Alpha;

        % Accept either numeric matrix or table; for table align columns by PredictorNames.
        Xq = ClassificationGP.toMatrixPredictors(Xnew, this.PredictorNames);

        % Apply training standardization parameters
        XqStd = (Xq - this.PredictorLocation) ./ this.PredictorScale;

        % Get active set vectors used during training
        XaStd = this.ActiveSetVectors(this.IsActiveSetVector,:);

        % Cross-covariances K(X_active, X_new) and diagonal of K(X_new, X_new)
        args = this.ModelParameters;
        Kxs = ClassificationGP.kernelMatrix(XaStd, XqStd, args) + abs(this.Lambda).*speye(size(XaStd,1),size(XqStd,1));
        Kss = ClassificationGP.kernelDiag(XqStd, args);

        % Mean function m(x)=H(x)*Beta
        Hq = ClassificationGP.basisMatrix(XqStd, this.BasisFunction);
        mq = Hq * this.Beta;

        switch lower(this.Inference)
            case 'ep'
                [pPos, pLo, pHi] = ClassificationGP.predictProbEP(this.Posterior_, Kxs, Kss, mq, alphaLevel);
            otherwise
                [pPos, pLo, pHi] = ClassificationGP.predictProbLaplace(this.Posterior_, Kxs, Kss, mq, alphaLevel, this.Link_, this.ProbitScaling_);
        end

        % Optional user transformation of probability outputs
        if ~isempty(this.ResponseTransform)
            pPos = this.ResponseTransform(pPos);
            pPos = min(max(pPos,0),1);
            pLo = this.ResponseTransform(pLo);
            pLo = min(max(pLo,0),1);
            pHi = this.ResponseTransform(pHi);
            pHi = min(max(pHi,0),1);
        end

        % Return in fitc-style: 2 columns aligned to Classes_ order
        score = [1-pPos(:), pPos(:)];
        label = repmat(this.Classes_(1), numel(pPos), 1);
        label(pPos >= 0.5) = this.Classes_(2);

        % Optional confidence interval (approximate)
        ci = [];
        if nargout > 2
            ci = [pLo, pHi];
        end
    end

    function L = loss(this, X, Y, varargin)
        %LOSS Compute classification loss on (X,Y).
        %
        % Name-value:
        %   'LossFun':
        %       - 'classiferror' (default)
        %       - 'hinge'
        %       - 'quadratic'
        %       - 'negloglikelihood'
        %       - function handle: @(Ytrue,Ypred,SCORE)->scalar

        ClassificationGP.validateInputs(X, Y);

        ip = inputParser;
        ip.addParameter('LossFun', 'classiferror');
        ip.addParameter('Weights', ones(size(Y)));
        ip.parse(varargin{:});
        lossFun = ip.Results.LossFun;
        w = ip.Results.Weights;

        % Validate Y against trained classes (allow partial coverage in test set)
        [yTrue, w] = ClassificationGP.normalizeY(Y, w, this.Classes_);
        yTrueLabel = yTrue > 0.5;

        [labels, score] = this.predict(X);
        yPredLabel = (labels == this.Classes_(2));
        yPredProb = score(:,2);

        if isa(lossFun,'function_handle')
            L = lossFun(yTrue, yPred, score);
            return;
        end

        switch lower(string(lossFun))
            case "classiferror"
                % 0/1 Loss (Misclassification Rate)
                L = sum(w.*double(yPredLabel ~= yTrueLabel));
            case "hinge"
                % Hinge loss usually requires class labels to be -1/+1
                % Mapping: 0->-1, 1->1.
                L = sum(w.*max(0,1 - (2*double(yTrueLabel) - 1).*(2.*yPredProb - 1)));
            case "quadratic"
                % Least squares
                L = sum(w.*(yPredProb - yTrue).^2);
            case "negloglikelihood"
                % Negative Log Likelihood
                L = -sum(w.*(yTrue.*log(yPredProb) + (1 - yTrue).*log(1 - yPredProb)));
            otherwise
                error('ClassificationGP:BadLossFun', 'Unsupported LossFun: %s', string(lossFun));
        end
    end

    function L = resubLoss(this, varargin)
        %RESUBLOSS Resubstitution loss (evaluate on training data).
        if ~isempty(this.X) && ~isempty(this.Y)
            L = this.loss(this.X, this.Y, varargin{:});
        else
            error('ClassificationGP:Compact', 'Unsupported for compacted model');
        end
    end

    function [label, score, ci] = resubPredict(this, varargin)
        %RESUBPREDICT Resubstitution predictions (predict on training data).
        if ~isempty(this.X)
            [label, score, ci] = this.predict(this.X, varargin{:});
        else
            error('ClassificationGP:Compact', 'Unsupported for compacted model');
        end
    end
end

methods (Static, Access=private)
    function validateInputs(X, Y)
        % Validate input data for GP classification

        % Check X
        if istable(X)
            X_numeric = table2array(X);
        else
            X_numeric = X;
        end

        if isempty(X_numeric)
            error('ClassificationGP:EmptyX', 'Input X cannot be empty.');
        end

        if ~isnumeric(X_numeric) && ~islogical(X_numeric)
            error('ClassificationGP:InvalidX', 'X must be numeric or logical.');
        end

        if any(isnan(X_numeric(:))) || any(isinf(X_numeric(:)))
            error('ClassificationGP:InvalidX', 'X contains NaN or Inf values.');
        end

        % Check Y
        if isempty(Y)
            error('ClassificationGP:EmptyY', 'Response Y cannot be empty.');
        end

        % Check size compatibility
        if size(X_numeric, 1) ~= size(Y,1)
            error('ClassificationGP:SizeMismatch', ...
                'Number of rows in X (%d) must equal rows of Y (%d).', ...
                size(X_numeric, 1), size(Y,1));
        end
    end

    function validateKernelParameters(theta, kernelFunction, d)
        % Validate that kernel parameters are appropriate

        kf = lower(string(kernelFunction));

        % Expected parameter counts
        if startsWith(kf, "ard")
            if contains(kf, "rationalquadratic")
                expected = d + 2; % [ell1,...,elld, sf, alpha]
            else
                expected = d + 1; % [ell1,...,elld, sf]
            end
        else
            if contains(kf, "rationalquadratic")
                expected = 3; % [ell, sf, alpha]
            else
                expected = 2; % [ell, sf]
            end
        end

        if numel(theta) ~= expected
            % Warning only, as some custom calls might differ
            warning('ClassificationGP:KernelParamMismatch', ...
                'Kernel ''%s'' with d=%d typically requires %d parameters, got %d.', ...
                kf, d, expected, numel(theta));
        end

        % All parameters must be positive
        if any(theta <= 0)
            error('ClassificationGP:BadKernelParams', ...
                'All kernel parameters must be positive.');
        end

        % Warn about extreme values
        if any(theta < 1e-6) || any(theta > 1e6)
            warning('ClassificationGP:ExtremeKernelParams', ...
                'Kernel parameters contain extreme values (< 1e-6 or > 1e6). This may cause numerical issues.');
        end
    end

    function args = parseFitArgs(X, Y, varargin)
        % Parse name/value pairs and create fitrgp-like defaults.

        ip = inputParser;
        ip.FunctionName = 'fitcgp';

        % ----- Model structure -----
        ip.addParameter('KernelFunction', 'squaredexponential');
        ip.addParameter('KernelParameters', [], @(v) isempty(v) || isnumeric(v));
        ip.addParameter('ConstantKernelParameters', [], @(v) isempty(v) || islogical(v));
        ip.addParameter('DistanceMethod', 'fast', @(s) any(strcmpi(string(s), ["fast","accurate"])));
        ip.addParameter('BasisFunction', 'constant');
        ip.addParameter('Beta', [], @(v) isempty(v) || isnumeric(v));

        % Lambda
        ip.addParameter('Lambda', 0, @(v) isempty(v) || (isnumeric(v) && isscalar(v) && v>0));
        ip.addParameter('ConstantLambda', true, @(b) islogical(b) && isscalar(b));

        % ----- Inference choice (user-facing) -----
        % Requested mapping:
        %   'Logit'  => Laplace + logit
        %   'Probit' => Laplace + probit
        %   'EP'     => EP + probit
        ip.addParameter('Inference', 'Logit', @(s) ischar(s) || isstring(s));
        ip.addParameter('InferenceOptions', struct(), @(s) isstruct(s));
        ip.addParameter('ProbitScaling', 'none', @(s) any(strcmpi(string(s), ["none","slope","minimax"])));

        % ----- Approximation / compute knobs -----
        ip.addParameter('FitMethod', [], @(s) isempty(s) || ischar(s) || isstring(s));
        ip.addParameter('PredictMethod', [], @(s) isempty(s) || ischar(s) || isstring(s));
        ip.addParameter('ActiveSet', [], @(v) isempty(v) || islogical(v) || isnumeric(v));
        ip.addParameter('ActiveSetSize', [], @(v) isempty(v) || (isnumeric(v) && isscalar(v) && v>=1));
        ip.addParameter('ActiveSetMethod', [], @(s) isempty(s) || ischar(s) || isstring(s));

        % Standardize predictors (centering/scaling)
        ip.addParameter('Standardize', false, @(b) islogical(b) || isscalar(b));

        % Diagnostics
        ip.addParameter('Verbose', 0, @(v) isnumeric(v) && isscalar(v));

        % ----- Local optimizer (fitrgp-like) -----
        ip.addParameter('Optimizer', 'quasinewton', @(s) ischar(s) || isstring(s));
        ip.addParameter('OptimizerOptions', [], @(o) isempty(o) || isstruct(o) || isobject(o));
        ip.addParameter('InitialStepSize', [], @(v) isempty(v) || isnumeric(v) || ischar(v) || isstring(v));

        % ----- Names / transforms -----
        ip.addParameter('PredictorNames', [], @(v) isempty(v) || iscellstr(v) || isstring(v));
        ip.addParameter('ResponseName', 'Y', @(s) ischar(s) || isstring(s));
        ip.addParameter('ResponseTransform', [], @(f) isempty(f) || isa(f,'function_handle'));

        % ----- bayesopt hyperparameter optimization -----
        ip.addParameter('OptimizeHyperparameters', 'none');
        ip.addParameter('HyperparameterOptimizationOptions', struct(), @(s) isstruct(s));

        % Optional weights (common in classifiers even if not in prompt list)
        ip.addParameter('Weights', [], @(v) isempty(v) || (isnumeric(v) && isvector(v)));

        ip.parse(varargin{:});
        args = ip.Results;
        args.X = X;
        args.Y = Y;

        % Convert standardize to logical
        args.Standardize = logical(args.Standardize);

        % Default FitMethod/PredictMethod heuristic similar to fitrgp
        n = size(ClassificationGP.peekN(X), 1);
        if isempty(args.FitMethod)
            args.FitMethod = 'exact';
        end
        if isempty(args.PredictMethod)
            args.PredictMethod = 'exact';
        end

        % Default active set size/method
        if isempty(args.ActiveSetSize)
            args.ActiveSetSize = min(ClassificationGP.DEFAULT_ACTIVE_SET_SIZE, n);
        end
        if isempty(args.ActiveSetMethod)
            args.ActiveSetMethod = 'random';
        end

        % Inference mapping requested in prompt
        infStr = lower(string(args.Inference));
        if infStr == "ep"
            args.InferenceMethod = 'EP';
            args.Likelihood = 'probit';
        elseif infStr == "probit"
            args.InferenceMethod = 'Laplace';
            args.Likelihood = 'probit';
        else
            args.InferenceMethod = 'Laplace';
            args.Likelihood = 'logit';
        end

        % Default Lambda
        if isempty(args.Lambda)
            args.Lambda = 0;
        end

        % Inference iteration defaults
        opt = args.InferenceOptions;
        if ~isfield(opt,'MaxIter')
            opt.MaxIter = ClassificationGP.DEFAULT_MAX_ITER;
        end
        if ~isfield(opt,'Tol')
            opt.Tol = ClassificationGP.DEFAULT_TOL;
        end
        if ~isfield(opt,'Damping')
            opt.Damping = ClassificationGP.DEFAULT_DAMPING;
        end
        args.InferenceOptions = opt;
    end

    function [Xmat, y01, w, meta] = prepareData(args)
        % Convert X to numeric matrix, determine names, normalize Y to binary.
        X = args.X;
        Y = args.Y;

        meta = struct();
        if istable(X)
            % If Y is a variable name, pull it from the table.
            if (ischar(Y) || isstring(Y)) && isscalar(string(Y))
                yName = string(Y);
                meta.ResponseName = yName;
                yVec = X.(yName);

                Xtbl = X;
                Xtbl.(yName) = [];

                Xmat = table2array(Xtbl);
                if isempty(args.PredictorNames)
                    meta.PredictorNames = string(Xtbl.Properties.VariableNames);
                else
                    meta.PredictorNames = string(args.PredictorNames);
                end
                meta.OriginalY = yVec;
            else
                Xmat = table2array(X);
                if isempty(args.PredictorNames)
                    meta.PredictorNames = string(X.Properties.VariableNames);
                else
                    meta.PredictorNames = string(args.PredictorNames);
                end
                meta.ResponseName = string(args.ResponseName);
                meta.OriginalY = Y;
            end
        else
            % Matrix input
            Xmat = X;
            meta.ResponseName = string(args.ResponseName);
            if isempty(args.PredictorNames)
                p = size(Xmat,2);
                meta.PredictorNames = "x" + (1:p);
            else
                meta.PredictorNames = string(args.PredictorNames);
            end
            meta.OriginalY = Y;
        end

        if isempty(args.Weights)
            w = ones(size(Xmat,1),1);
        else
            w = args.Weights(:);
        end

        [y01, w, classes] = ClassificationGP.normalizeY(meta.OriginalY, w, []);
        meta.Classes = classes;
    end

    function [y01, w, classes, yCat] = normalizeY(Y, w, expectedClasses)
        % Normalize Y to categorical with 2 classes.
        % expectedClasses: Optional. If provided, Y is validated against it.

        % Convert input to categorical first
        if isnumeric(Y) && size(Y,2)==2
            % Binomial input
            p = Y(:,1)./Y(:,2)
            w = w.*Y(:,2);
            y = p >= 0.5;
            y01 = p;
            yCat = categorical(y, [false true], {'0','1'});
        elseif isnumeric(Y) && isvector(Y)
            yv = Y(:);
            if all((yv>=0) & (yv<=1))
                % Probability input
                y = yv >= 0.5;
                y01 = yv;
                yCat = categorical(y, [false true], {'0','1'});
            else
                % Numeric labels input
                u = sort(unique(yv));
                if numel(u) > 2
                    error('ClassificationGP:BinaryOnly', 'Numeric Y must have exactly 2 unique values.');
                elseif numel(u) < 2
                    y01 = (yv == u(1));
                    yCat = categorical(y01, [false true], [num2str(0),num2str(u(1))]);
                else
                    y01 = (yv == u(2));
                    yCat = categorical(y01, [false true], [num2str(u(1)),num2str(u(2))]);
                end
            end
        elseif islogical(Y)
            % Logical input
            yv = Y(:);
            yCat = categorical(yv, [false true], {'0','1'});
            y01 = double(yv);
        elseif iscategorical(Y)
            % Categorical input
            yCat = Y(:);
            cats = categories(yCat,'OutputType','string');
            if length(cats) ~= 2
                yCat = removecats(yCat);
                cats = categories(yCat,'OutputType','string');
            end
            if length(cats) ~= 2
                error('ClassificationGP:BinaryOnly', 'Training data must have exactly 2 classes.');
            end
            y01 = (yCat == cats(2));
        else
            error('ClassificationGP:BadY', 'Unsupported Y type.');
        end

        cats = categories(yCat);
        if ~isempty(expectedClasses)
            % Validate against model classes
            if ~all(ismember(cats, categories(expectedClasses)))
                error('ClassificationGP:UnknownClass', 'Y contains classes not in the training set.');
            end
            classes = expectedClasses;
            % Ensure yCat has the same categories/order
            yCat = categorical(yCat, categories(expectedClasses));
        else
            classes = categorical(cats);
        end
    end

    function Xq = toMatrixPredictors(Xnew, predictorNames)
        % Convert predictors to matrix. For tables, respect PredictorNames order.
        if istable(Xnew)
            Xq = table2array(Xnew(:, cellstr(predictorNames)));
        else
            Xq = Xnew;
        end
    end

    function [Xs, mu, sig] = standardizeX(X)
        % Standardize each predictor column to zero mean and unit std.
        mu = mean(X,1,'omitnan');
        sig = std(X,0,1,'omitnan');
        sig(sig==0) = 1;
        Xs = (X - mu) ./ sig;
    end

    function beta = ensureBetaSize(beta, p)
        % Ensure Beta is the correct length given basis matrix size.
        if isempty(beta)
            beta = zeros(p,1);
        else
            beta = beta(:);
            if numel(beta) ~= p
                error('ClassificationGP:BadBeta', 'Beta must be %d-by-1.', p);
            end
        end
    end

    function H = basisMatrix(X, basis)
        % Construct explicit basis matrix H given BasisFunction.
        if isa(basis,'function_handle')
            H = basis(X);
            return;
        end
        b = lower(string(basis));
        n = size(X,1);
        switch b
            case "none"
                H = zeros(n,0);
            case "constant"
                H = ones(n,1);
            case "linear"
                H = [ones(n,1), X];
            case "purequadratic"
                H = [ones(n,1), X, X.^2];
            otherwise
                error('ClassificationGP:BadBasis', 'Unsupported BasisFunction: %s', b);
        end
    end

    function [Xa, ya, wa, isActive] = selectActiveSet(X, y01, w, args)
        % Choose active set indices for compute control.
        %
        % User can supply `ActiveSet` explicitly. Otherwise we choose M points
        % according to `ActiveSetMethod`.

        n = size(X,1);

        if ~isempty(args.ActiveSet)
            as = args.ActiveSet;

            if islogical(as) || all(ismember(vec, [0 1]))
                isActive = logical(as(:));
                if (size(X,1) >= size(isActive,1))
                    error('ClassificationGP:ActiveSetSizeMismatch', ...
                        'Number of rows in ActiveSet (%d) must equal rows of X (%d).', ...
                        size(isActive,1), size(X,1));
                end
            else
                as = round(as);
                if (size(X,1) <= size(as,1))
                    error('ClassificationGP:ActiveSetSizeMismatch', ...
                        'Number of rows in ActiveSet (%d) must be lower or equal X (%d).', ...
                        size(as,1), size(X,1));
                end
                if (numel(as) == numel(unique(as)))
                    error('ClassificationGP:ActiveSetNotUnique', ...
                        'ActiveSet must contain unique numbers.');
                end
                if (max(as) <= n)
                    error('ClassificationGP:ActiveSetBadIdxs', ...
                        'ActiveSet indices must be lower or equal rows of X (%d)', ...
                        max(as), size(X,1));
                end
                if (any(as <= 0))
                    error('ClassificationGP:ActiveSetBadIdxs', ...
                        'ActiveSet indices must positive integers');
                end
                isActive = false(n,1);
                isActive(as) = true;
            end
        else
            m = min(args.ActiveSetSize, n);
            method = lower(string(args.ActiveSetMethod));
            switch method
                case "first"
                    idx = (1:m)';
                case "last"
                    idx = (n-m+1:n)';
                case "random"
                    idx = randperm(n, m)';
                case "qr"
                    % QR selection
                    Xc = X - mean(X,1);
                    [~,~,p] = qr(Xc','econ','vector');
                    idx = p(1:m);
                otherwise
                    error('ClassificationGP:BadActiveSetMethod', 'Unsupported ActiveSetMethod: %s', method);
            end
            isActive = false(n,1);
            isActive(idx) = true;
        end

        Xa = X(isActive,:);
        ya = y01(isActive,:);
        wa = w(isActive,:);
    end

    function argsOut = optimizeLocally(argsIn, XaStd, ya01, wa, Ha)
        % Local hyperparameter optimization.
        %
        % We maximize the approximate marginal log likelihood produced by
        % Laplace/EP. This is analogous to fitrgp's evidence optimization, but
        % uses our approximate inference objective instead of exact GPR evidence.

        argsOut = argsIn;

        [x0, packInfo] = ClassificationGP.packFreeParameters(argsIn, Ha);
        if isempty(x0)
            return;
        end

        obj = @(x) ClassificationGP.negLogEvidenceFromFree(x, packInfo, argsIn, XaStd, ya01, wa, Ha);

        optName = lower(string(argsIn.Optimizer));
        [xBest, fBest] = ClassificationGP.runLocalOptimizer(obj, x0, optName, argsIn.OptimizerOptions, argsIn.Verbose);

        argsOut = ClassificationGP.unpackFreeParameters(xBest, packInfo, argsIn, Ha);

        % If optimization looks failed, keep initial args
        if ~isfinite(fBest)
            argsOut = argsIn;
        end
    end

    function [x0, info] = packFreeParameters(args, H)
        % Pack model parameters into an unconstrained vector for optimization.
        %
        % Conventions used:
        %   - Positive parameters (kernel parameters, Lambda) are optimized in log space.
        %   - Beta is optimized in linear space.
        %
        % Constant parameters:
        %   - ConstantKernelParameters excludes kernel params from optimization.
        %   - ConstantLambda keeps Lambda fixed.

        theta = args.KernelParameters;
        if isempty(theta)
            theta = ClassificationGP.defaultKernelParameters(args.KernelFunction, size(args.X,2));
        end
        theta = theta(:);

        ck = args.ConstantKernelParameters;
        if isempty(ck)
            ck = false(numel(theta),1);
        else
            ck = ck(:);
            if numel(ck) ~= numel(theta)
                error('ClassificationGP:BadConstantKernelParameters', ...
                    'ConstantKernelParameters must match length of KernelParameters.');
            end
        end

        freeThetaIdx = find(~ck);
        xTheta0 = log(theta(freeThetaIdx));

        beta0 = ClassificationGP.ensureBetaSize(args.Beta, size(H,2));
        xBeta0 = beta0(:);

        if args.ConstantLambda
            xLambda0 = [];
        else
            xLambda0 = log(args.Lambda);
        end

        x0 = [xTheta0; xBeta0; xLambda0];

        info = struct();
        info.theta0 = theta;
        info.freeThetaIdx = freeThetaIdx;
        info.betaSize = size(H,2);
        info.hasLambda = ~args.ConstantLambda;
    end

    function args = unpackFreeParameters(x, info, args, H)
        % Unpack unconstrained vector back into structured args.
        theta = info.theta0;
        k = numel(info.freeThetaIdx);
        if k > 0
            theta(info.freeThetaIdx) = exp(x(1:k));
        end

        betaStart = k + 1;
        betaEnd = betaStart + info.betaSize - 1;
        beta = x(betaStart:betaEnd);

        if info.hasLambda
            lambdaRaw = x(betaEnd+1);
            lambda = exp(lambdaRaw);
        else
            lambda = args.Lambda;
        end

        args.KernelParameters = theta;
        args.Beta = beta(:);
        args.Lambda = lambda;

        ClassificationGP.ensureBetaSize(args.Beta, size(H,2));
    end

    function nle = negLogEvidenceFromFree(x, info, args0, XaStd, ya01, wa, Ha)
        % Objective = negative approximate evidence.
        args = ClassificationGP.unpackFreeParameters(x, info, args0, Ha);
        beta = args.Beta(:);

        [~, ll] = ClassificationGP.runInferenceFromArgs(args, XaStd, ya01, wa, Ha, beta);
        if ~isfinite(ll)
            nle = 1e10;
        else
            nle = -ll;
        end
    end

    function [xBest, fBest] = runLocalOptimizer(obj, x0, optName, optOptions, verbose)
        % Run a chosen local optimizer. Falls back gracefully when toolboxes are absent.

        switch verbose
            case 0
                disp = 'off';
            case 1
                disp = 'iter';
            otherwise
                disp = 'iter-detailed';
        end

        switch optName
            case "fminsearch"
                if isempty(optOptions)
                    optOptions = optimset('Display',disp);
                end
                [xBest, fBest] = fminsearch(obj, x0, optOptions);

            case {"quasinewton","lbfgs","fminunc"}
                if exist('fminunc','file') ~= 2
                    error('ClassificationGP:NoFminunc', 'Optimizer fminunc requires Optimization Toolbox.');
                end
                if isempty(optOptions)
                    optOptions = optimoptions('fminunc', 'Algorithm','quasi-newton', 'Display',disp);
                else
                    try
                        if ~isa(optOptions,'optim.options.Fminunc') && isstruct(optOptions)
                            optOptions = optimoptions('fminunc', optOptions);
                        end
                    catch
                    end
                end
                [xBest, fBest] = fminunc(obj, x0, optOptions);

            case "fmincon"
                if exist('fmincon','file') ~= 2
                    error('ClassificationGP:NoFmincon', 'Optimizer ''fmincon'' requires Optimization Toolbox.');
                end
                if isempty(optOptions)
                    optOptions = optimoptions('fmincon','Display',disp);
                else
                    try
                        if ~isa(optOptions,'optim.options.Fmincon') && isstruct(optOptions)
                            optOptions = optimoptions('fmincon', optOptions);
                        end
                    catch
                    end
                end
                [xBest, fBest] = fmincon(obj, x0, [],[],[],[],[],[],[], optOptions);

            otherwise
                error('ClassificationGP:BadOptimizer', 'Unsupported Optimizer: %s', optName);
        end
    end

    function [post, ll] = runInferenceFromArgs(args, XaStd, ya01, wa, Ha, beta)
        % Build K and m, then run Laplace or EP inference.
        m = Ha * beta;
        K = ClassificationGP.kernelMatrix(XaStd, XaStd, args) + abs(args.Lambda).*speye(size(XaStd,1));

        switch lower(args.InferenceMethod)
            case 'laplace'
                [post, ll] = ClassificationGP.inferLaplace(K, ya01, wa, m, args.Likelihood, args.ProbitScaling, args.InferenceOptions);
            case 'ep'
                if ~strcmpi(args.Likelihood,'probit')
                    error('ClassificationGP:EPRequiresProbit', 'EP inference requires Probit.');
                end
                [post, ll] = ClassificationGP.inferEP(K, ya01, wa, m, args.InferenceOptions);
            otherwise
                error('ClassificationGP:BadInference', 'InferenceMethod must be Laplace or EP.');
        end
    end

    function mp = buildModelParameters(args, this)
        % Build a struct summarizing fitted configuration (fitrgp-like property).
        mp = struct( ...
            'KernelFunction', args.KernelFunction, ...
            'KernelParameters', args.KernelParameters, ...
            'BasisFunction', args.BasisFunction, ...
            'Beta', this.Beta, ...
            'Lambda', this.Lambda, ...
            'Inference', this.Inference, ...
            'Likelihood', this.Link_, ...
            'Standardize', this.Standardize_, ...
            'FitMethod', this.FitMethod, ...
            'PredictMethod', this.PredictMethod, ...
            'ActiveSetSize', this.ActiveSetSize, ...
            'ActiveSetMethod', this.ActiveSetMethod, ...
            'DistanceMethod', this.DistanceMethod_, ...
            'ProbitScaling', this.ProbitScaling_, ...
            'Optimizer', args.Optimizer, ...
            'OptimizerOptions', args.OptimizerOptions, ...
            'OptimizeHyperparameters', args.OptimizeHyperparameters, ...
            'HyperparameterOptimizationOptions', args.HyperparameterOptimizationOptions, ...
            'ConstantLambda', args.ConstantLambda, ...
            'ConstantKernelParameters', args.ConstantKernelParameters);
    end

    function [bestArgs, results] = runBayesopt(Xraw, y01, w, ~, args0)
        % Run bayesopt to minimize negative approximate evidence.
        %
        % Each bayesopt evaluation:
        %   - applies candidate hyperparameters to args
        %   - builds standardized/active-set data
        %   - optionally runs local optimization
        %   - returns Objective = -LogEvidence

        if exist('bayesopt','file') ~= 2
            error('ClassificationGP:NoBayesopt', 'OptimizeHyperparameters requires bayesopt.');
        end

        hyp = string(args0.OptimizeHyperparameters);

        if strcmpi(hyp, "auto")
            hypList = ["KernelScale","Standardize"];
        elseif strcmpi(hyp, "all")
            hypList = ["BasisFunction","KernelFunction","KernelScale","Standardize"];
        elseif isa(args0.OptimizeHyperparameters,'optimizableVariable')
            vars = args0.OptimizeHyperparameters;
            hypList = string({vars.Name});
        else
            hypList = string(args0.OptimizeHyperparameters);
            if isscalar(hypList) && strcmpi(hypList,"none")
                bestArgs = args0;
                results = [];
                return;
            end
        end

        if ~exist('vars','var')
            vars = ClassificationGP.defaultOptimizableVariables(hypList);
        end

        bo = args0.HyperparameterOptimizationOptions;
        if ~isfield(bo,'MaxObjectiveEvaluations')
            bo.MaxObjectiveEvaluations = 30;
        end
        if ~isfield(bo,'Verbose')
            bo.Verbose = 0;
        end
        if ~isfield(bo,'IsObjectiveDeterministic')
            bo.IsObjectiveDeterministic = true;
        end
        if ~isfield(bo,'AcquisitionFunctionName')
            bo.AcquisitionFunctionName = 'expected-improvement-plus';
        end

        objFcn = @(T) ClassificationGP.bayesObjective(T, Xraw, y01, w, args0);

        resultsBO = bayesopt(objFcn, vars, ...
            'MaxObjectiveEvaluations', bo.MaxObjectiveEvaluations, ...
            'Verbose', bo.Verbose, ...
            'IsObjectiveDeterministic', bo.IsObjectiveDeterministic, ...
            'AcquisitionFunctionName', bo.AcquisitionFunctionName);

        bestT = resultsBO.XAtMinObjective;
        bestArgs = ClassificationGP.applyBayesVarsToArgs(bestT, args0);

        results = struct();
        results.OptimizationResults = resultsBO;
        results.XAtMinObjective = bestT;
        results.MinObjective = resultsBO.MinObjective;
    end

    function vars = defaultOptimizableVariables(hypList)
        % Default hyperparameter search space (fitrgp-like spirit).
        hypList = string(hypList);
        vars = optimizableVariable.empty(0,1);

        if any(hypList=="Standardize")
            vars(end+1) = optimizableVariable('Standardize', [0 1], 'Type','integer');
        end
        if any(hypList=="Lambda")
            vars(end+1) = optimizableVariable('Lambda', [ClassificationGP.MIN_JITTER ClassificationGP.MAX_JITTER], 'Transform','log');
        end
        if any(hypList=="KernelFunction")
            vars(end+1) = optimizableVariable('KernelFunction', ...
                ["squaredexponential","exponential","matern32","matern52","rationalquadratic"], ...
                'Type','categorical');
        end
        if any(hypList=="BasisFunction")
            vars(end+1) = optimizableVariable('BasisFunction', ...
                ["none","constant","linear","purequadratic"], ...
                'Type','categorical');
        end
        if any(hypList=="KernelScale")
            vars(end+1) = optimizableVariable('KernelScale', [1e-3 1e3], 'Transform','log');
        end
    end

    function obj = bayesObjective(T, Xraw, y01, w, args0)
        % bayesopt ObjectiveFunction:
        % return table with variable 'Objective' to minimize.

        args = ClassificationGP.applyBayesVarsToArgs(T, args0);

        if args.Standardize
            Xstd = ClassificationGP.standardizeX(Xraw);
        else
            Xstd = Xraw;
        end

        [XaStd, ya01, wa] = ClassificationGP.selectActiveSet(Xstd, y01, w, args);

        Ha = ClassificationGP.basisMatrix(XaStd, args.BasisFunction);
        args.Beta = ClassificationGP.ensureBetaSize(args.Beta, size(Ha,2));

        % Like fitrgp, allow local optimization inside each bayesopt trial.
        if ~strcmpi(string(args.FitMethod), "none")
            args = ClassificationGP.optimizeLocally(args, XaStd, ya01, wa, Ha);
        end

        beta = args.Beta(:);
        [~, ll] = ClassificationGP.runInferenceFromArgs(args, XaStd, ya01, wa, Ha, beta);
        if ~isfinite(ll)
            obj = 1e10;
        else
            obj = -ll;
        end
    end

    function args = applyBayesVarsToArgs(T, args)
        % Apply bayesopt-selected variables into args struct.
        varNames = T.Properties.VariableNames;

        if ismember('Standardize', varNames)
            args.Standardize = logical(T.Standardize);
        end
        if ismember('Lambda', varNames)
            args.Lambda = T.Lambda;
            args.ConstantLambda = true; % treat as fixed during local optimization
        end
        if ismember('KernelFunction', varNames)
            args.KernelFunction = char(string(T.KernelFunction));
            args.KernelParameters = [];
            args.ConstantKernelParameters = [];
        end
        if ismember('BasisFunction', varNames)
            args.BasisFunction = char(string(T.BasisFunction));
            args.Beta = [];
        end
        if ismember('KernelScale', varNames)
            % Interpret as a shared length scale for non-ARD kernels.
            kf = lower(string(args.KernelFunction));
            if startsWith(kf,"ard")
                error('ClassificationGP:KernelScaleNotForARD', ...
                    'KernelScale cannot be optimized for ARD kernels.');
            end

            ell = T.KernelScale;
            theta = args.KernelParameters;
            if isempty(theta)
                theta = ClassificationGP.defaultKernelParameters(args.KernelFunction, size(args.X,2));
            end
            theta = theta(:);
            theta(1) = ell;
            args.KernelParameters = theta;

            ck = args.ConstantKernelParameters;
            if isempty(ck)
                ck = false(numel(theta),1);
            else
                ck = ck(:);
            end
            ck(1) = true;
            args.ConstantKernelParameters = ck;
        end
    end

    function theta = defaultKernelParameters(kernelFunction, d)
        % Default kernel parameters used when user doesn't provide initial values.
        kf = lower(string(kernelFunction));
        if startsWith(kf,"ard")
            theta = [ones(d,1); 1];
        else
            theta = [1; 1];
        end
        if contains(kf,"rationalquadratic")
            theta = [theta; 1];
        end
    end

    function K = kernelMatrix(X1, X2, args)
        % Evaluate covariance matrix K(X1,X2) for supported kernels.
        %
        % For built-in string kernels, we implement standard stationary kernels.
        % If KernelFunction is a function handle, we call:
        %   K = kf(X1,X2,theta)

        kf = args.KernelFunction;
        theta = args.KernelParameters;

        if isempty(theta)
            theta = ClassificationGP.defaultKernelParameters(kf, size(X1,2));
        end
        theta = theta(:);

        if isa(kf,'function_handle')
            K = kf(X1, X2, theta);
            return;
        end

        name = lower(string(kf));
        dist = lower(string(args.DistanceMethod));

        % Optimized ARD/Iso selection
        if startsWith(name,"ard")
            d = size(X1,2);
            ell = theta(1:d)';
            sf = theta(d+1);
            base = erase(name,"ard");
            % Use pdist2 with weights (1./ell)
            if strcmpi(dist,"accurate")
                R = pdist2(X1./ell, X2./ell);
            else
                R = pdist2(X1./ell, X2./ell,'fasteuclidean');
            end
        else
            ell = theta(1);
            sf = theta(2);
            % Use pdist2 with weights (1./ell)
            if strcmpi(dist,"accurate")
                R = pdist2(X1./ell, X2./ell);
            else
                R = pdist2(X1./ell, X2./ell,'fasteuclidean');
            end
            base = name;
        end
        K = ClassificationGP.kernelFromR(base, R, sf, theta);
    end

    function kdiag = kernelDiag(X, args)
        % Return diag(K(X,X)) efficiently for stationary kernels.
        kf = args.KernelFunction;
        theta = args.KernelParameters;

        if isempty(theta)
            theta = ClassificationGP.defaultKernelParameters(kf, size(X,2));
        end
        theta = theta(:);

        if isa(kf,'function_handle')
            K = kf(X, X, theta);
            kdiag = diag(K);
            return;
        end

        name = lower(string(kf));
        d = size(X,2);
        if startsWith(name,"ard")
            sf = theta(d+1);
        else
            sf = theta(2);
        end
        kdiag = (sf^2) * ones(size(X,1),1);
    end

    function K = kernelFromR(name, R, sf, theta)
        % Build K given distance matrix R and kernel hyperparameters.
        sf2 = sf^2;
        switch lower(string(name))
            case "squaredexponential"
                K = sf2 * exp(-0.5 * R.^2);
            case "exponential"
                K = sf2 * exp(-R);
            case "matern32"
                a = sqrt(3)*R;
                K = sf2 * (1 + a) .* exp(-a);
            case "matern52"
                a = sqrt(5)*R;
                K = sf2 * (1 + a + (a.^2)/3) .* exp(-a);
            case "rationalquadratic"
                alpha = theta(end);
                K = sf2 * (1 + (R.^2)./(2*alpha)).^(-alpha);
            otherwise
                error('ClassificationGP:BadKernel', 'Unsupported kernel: %s', name);
        end
    end

    function [post, logZ] = inferLaplace(K, y01, w, m, link, scaling, opt)
        % inferLaplace - Laplace approximation for Binary GP Classification
        %
        % Methods:
        %   Laplace: Newton-Raphson with Backtracking Line Search
        %
        % Usages:
        %   [post, logZ] = inferLaplace(K, y, w, m, 'logit', [], opt) where
        %   logZ is the approximate marginal log likelihood (Laplace evidence).

        n = size(K,1);
        y = double(y01(:));
        w = w(:);
        if isempty(m)
            m = zeros(n,1);
        end

        % Initialize at prior mean
        f = m;

        % Newton Raphson with Line Search
        obj_old = -inf;
        alpha = zeros(n,1); % alpha = K\(f-m)
        for it = 1:opt.MaxIter
            % Compute Likelihood derivatives
            [logL, grad, W] = ClassificationGP.likelihoodMoments(f, y, w, link);

            % Objective
            obj = sum(logL) - 0.5.*(alpha'*(f - m));

            % Convergence check on objective (ascent check)
            if abs(obj - obj_old) < opt.Tol*max(1, abs(obj))
                break;
            end
            obj_old = obj;

            % Compute Newton Direction
            sW = sqrt(W);
            B = (sW.*(K.*sW')) + speye(n);
            L = ClassificationGP.cholSafe(B);

            % Newton Step
            b = W.*(f - m) + grad;

            % a_dir is the target alpha for the full Newton step
            a_dir = b - sW.*(L'\(L\(sW.*(K*b))));

            % Direction d_alpha = a_dir - alpha
            d_alpha = a_dir - alpha;
            d_f = K*d_alpha;

            % Backtracking Line Search
            step = 1.0;
            while (step > eps)
                % Propose new state
                alpha_new = alpha + step*d_alpha;
                f_new = f + step*d_f;

                % Evaluate Objective
                logL = ClassificationGP.likelihoodMoments(f_new, y, w, link);
                obj_new = sum(logL) - 0.5*(alpha_new'*(f_new - m));

                % Simple ascent check
                if obj_new > obj
                    f = f_new;
                    alpha = alpha_new;
                    break;
                end
                step = 0.5*step;
            end
        end

        if (it==opt.MaxIter)
            warning('ClassificationGP:MaxIterReached', ...
                'Maximum iterations reached without convergence');
        end

        % Final quantities
        sW = sqrt(W);
        B = (sW.*(K.*sW')) + speye(n);
        L = ClassificationGP.cholSafe(B);

        % Approximate log evidence
        quad = alpha'*(f - m);
        logdetB = 2*sum(log(diag(L)));
        logZ = sum(logL) - 0.5*quad - 0.5*logdetB;

        post = struct();
        post.f_hat = f;
        post.W = W;
        post.L = L;
        post.sW = sW;
        post.alpha = alpha;
        post.K = K;
        post.m = m;
        post.link = link;
        post.inference = 'laplace';
    end

    function [post, logZ] = inferEP(K, y01, w, m, opt)
        % inferEP - Expectation Propagation for Binary GP Classification.
        %
        %   EP: Sequential Expectation Propagation (EP) with Rank-1
        %   updates. EP approximates the posterior with a Gaussian by
        %   iteratively matching moments of "tilted" distributions.
        %
        % Usages:
        %   [post, logZ] = inferEP(K, y, w, m, opt) 

        n = size(K,1);
        y = double(y01(:));
        y = 2*(y > 0.5) - 1; % Convert to -1/+1
        if isempty(m)
            m = zeros(n,1);
        end

        % Initialize Sites
        tau = zeros(n,1);
        nu  = zeros(n,1);

        % Initialize Posterior (Sigma = K, mu = m)
        Sigma = K;
        mu = m;

        % Loop Sweeps
        sqrt_eps = sqrt(eps);
        for it = 1:opt.MaxIter
            tau_old_sweep = tau;
            nu_old_sweep = nu;
            
            % Randomize order for better convergence
            perm = 1:n;%randperm(n);

            for i = perm
                sig_i = Sigma(i,i);
                tau_i = tau(i);
                nu_i = nu(i);
                mu_i = mu(i);
                y_i = y(i);

                % Compute Cavity Distribution
                denom = 1 - sig_i*tau_i;
                if abs(denom) < sqrt_eps
                    denom = sqrt_eps*sign(denom); 
                end
                var_cav = max(sig_i/denom,sqrt_eps);
                mu_cav = (mu_i - sig_i*nu_i)/denom;

                % Moment Matching (Probit)
                denom_margin = sqrt(1 + var_cav);
                z = (y_i*mu_cav)/denom_margin;

                % Mills Ratio for stable update
                ratio = ClassificationGP.millsRatioSafe(z);
 
                % New moments
                mu_hat = mu_cav + (y_i*var_cav/denom_margin)*ratio;
                var_hat = var_cav - (var_cav^2/(1 + var_cav))*ratio.*(ratio + z);
                var_hat = max(var_hat,sqrt_eps);

                % Update Site Parameters
                delta_tau = (1/var_hat) - (1/var_cav) - tau_i;
                delta_nu  = (mu_hat/var_hat) - (mu_cav/var_cav) - nu_i;

                % Damping
                delta_tau = delta_tau*opt.Damping;
                delta_nu  = delta_nu*opt.Damping;

                tau_new = max(tau_i + delta_tau,0);
                nu_new  = nu_i + delta_nu;
                tau(i) = tau_new;
                nu(i)  = nu_new;

                % Recompute deltas based on clamped values
                d_tau = tau_new - tau_i;
                d_nu  = nu_new - nu_i;

                % Rank-1 Update of Posterior Sigma and mu
                si = Sigma(:,i);
                denom_update = 1 + d_tau*sig_i;

                % If update is too singular, skip it
                if abs(denom_update) > sqrt_eps
                    K_update_factor = d_tau/denom_update;
                    mu_update_factor = (d_nu - d_tau*mu_i)/denom_update;

                    Sigma = Sigma - K_update_factor*(si*si');
                    mu = mu + mu_update_factor*si;
                end
            end

            % Recompute Sigma globally to avoid numerical drift
            sW = sqrt(tau);
            B = (sW.*(K.*sW')) + speye(n);
            L = ClassificationGP.cholSafe(B);
            V = L\(K.*sW');
            Sigma = K - (V'*V);

            % Recompute Mu
            if norm(m) < sqrt_eps
                mu = Sigma*nu + (m - Sigma*(tau.*m));
            else
                LK = ClassificationGP.cholSafe(K);
                mu = Sigma * (nu + (LK')\(LK\m));
            end

            % Check Convergence
            if (max(abs(tau - tau_old_sweep)) < opt.Tol) && (max(abs(nu - nu_old_sweep)) < opt.Tol)
                break;
            end
        end

        if (it==opt.MaxIter)
            warning('ClassificationGP:MaxIterReached', ...
                'Maximum iterations reached without convergence');
        end

        % Compute Cavity Variances and Means
        denom = 1 - diag(Sigma).*tau;
        if abs(denom) < sqrt_eps
            denom = sqrt_eps*sign(denom);
        end
        var_cav = max(diag(Sigma)./denom,sqrt_eps);
        mu_cav = (mu - diag(Sigma).*nu)./denom;

        % Compute Moment Matching Normalization (Z_hat)
        denom_margin = sqrt(1 + var_cav);
        z = (y.*mu_cav)./denom_margin;
        logP = log(ClassificationGP.normcdfSafe(z));

        % Compute full LogZ
        sW = sqrt(tau);
        B = (sW.*(K.*sW')) + speye(n);
        L = ClassificationGP.cholSafe(B);
        logdetB = 2*sum(log(diag(L)));
        quad = nu'*(K*nu + 2*m); 
        logZ = sum(logP) - 0.5*quad -0.5*logdetB;

        LK = ClassificationGP.cholSafe(K);
        alpha = (LK')\(LK\(mu - m));

        post = struct();
        post.tau = tau;
        post.nu = nu;
        post.Sigma = Sigma;
        post.mu = mu;
        post.alpha = alpha;
        post.K = K;
        post.m = m;
        post.inference = 'ep';
    end

    function [logp, grad, W, p] = likelihoodMoments(f, y, w, link)
        % Compute likelihood moments for Laplace approximation.
        % Returns:
        %   logp: Log Likelihood values
        %   grad: Gradient of Log Likelihood w.r.t f
        %   W:    Diagonal of the expected Hessian (Fisher Information)
        %   p:    Probabilities

        f = f(:);
        y = y(:);
        w = w(:);

        switch lower(string(link))
            case "logit"
                % --- Logit (Logistic) Link ---
                p = ClassificationGP.sigmoidSafe(f);
                p = min(max(p, eps), 1-eps); % probability

                if nargout > 1
                    % Gradient
                    grad = w.*(y - p); % gradient
                end
                if nargout > 2
                    % Fisher Information
                    W = max(w.*p.*(1 - p),eps); 
                end
            case "probit"
                % --- Probit Link ---
                p = ClassificationGP.normcdfSafe(f); % probability
                p = min(max(p, eps), 1-eps);

                if nargout > 1
                    phi = ClassificationGP.normpdfSafe(f);
                    factor = p.*(1-p);

                    % Gradient
                    grad = w.*(phi./factor).*(y - p);
                end
                if nargout > 2
                    % Fisher Information
                    W = max(w.*(phi.^2./factor),0);
                end

            otherwise
                error('ClassificationGP:BadLikelihood', 'Unsupported likelihood: %s', string(link));
        end

        % Log Likelihood (Binomial/Bernoulli)
        logp = w.*(y.*log(p) + (1-y).*log(1-p));
    end

    function [pPos, pLo, pHi] = predictProbLaplace(post, Kxs, Kss, mq, alphaLevel, link, scaling)
        % Predictive probability using Laplace posterior.
        %
        % Predict latent mean:
        %   E[f*] ≈ m(x*) + K(Xa,x*)' * alpha
        %
        % Predict latent variance:
        %   Var[f*] ≈ k(x*,x*) - v'v
        % where v = L \ (sqrt(W) .* K(Xa,x*))

        mstar = mq(:) + (Kxs'*post.alpha);

        v = post.L\(post.sW.*Kxs);
        vLatent = max(Kss(:) - sum(v.^2,1)', 0);

        z = sqrt(2)*erfcinv(alphaLevel); % norminv(1-alpha/2)

        switch lower(string(link))
            case "probit"
                switch lower(string(scaling))
                    case "minimax"
                        sf = ClassificationGP.PROBIT_MINIMAX;
                    case "slope"
                        sf = ClassificationGP.PROBIT_SLOPE;
                    otherwise
                        sf = 1; % no scaling
                end

                denom = sqrt(1 + (sf^2)*vLatent)./sf;
                pPos = ClassificationGP.normcdfSafe(mstar./denom);
                if nargout > 1
                    mstarLo = mstar - z*sqrt(vLatent(:));
                    pLo = ClassificationGP.normcdfSafe(mstarLo./denom);
                end
                if nargout > 2
                    mstarHi = mstar + z*sqrt(vLatent(:));
                    pHi = ClassificationGP.normcdfSafe(mstarHi./denom);
                end
            case "logit"
                % optional variance correction
                switch lower(string(scaling))
                    case "minimax"
                        sf = ClassificationGP.PROBIT_MINIMAX;
                    case "slope"
                        sf = ClassificationGP.PROBIT_SLOPE;
                    otherwise
                        sf = 0; % no correction
                end

                denom = sqrt(1 + (sf^2)*vLatent);
                pPos = ClassificationGP.sigmoidSafe(mstar./denom);
                if nargout > 1
                    mstarLo = mstar - z*sqrt(vLatent(:));
                    pLo = ClassificationGP.sigmoidSafe(mstarLo./denom);
                end
                if nargout > 2
                    mstarHi = mstar + z*sqrt(vLatent(:));
                    pHi = ClassificationGP.sigmoidSafe(mstarHi./denom);
                end
            otherwise
                error('ClassificationGP:BadLikelihood', 'Unknown link.');
        end
    end

    function [pPos, pLo, pHi] = predictProbEP(post, Kxs, Kss, mq, alphaLevel)
        % Predictive probability using EP posterior (probit).
        %
        % Uses alpha-like mean term and an approximate latent variance computed
        % from EP covariance.

        mstar = mq(:) + (Kxs'*post.alpha);

        LK = ClassificationGP.cholSafe(post.K);
        KiKxs = LK'\(LK\Kxs);
        KiSigmaKiKxs = LK'\(LK\(post.Sigma * KiKxs));

        v = sum(Kxs.*(KiKxs - KiSigmaKiKxs), 1)';
        vLatent = max(Kss(:) - v, 0);

        z = sqrt(2)*erfcinv(alphaLevel); % norminv(1-alpha/2)

        denom = sqrt(1+vLatent);
        pPos = ClassificationGP.normcdfSafe(mstar ./ denom);
        if nargout > 1
            mstarLo = mstar - z*sqrt(vLatent(:));
            pLo = ClassificationGP.normcdfSafe(mstarLo ./ denom);
        end
        if nargout > 2
            mstarHi = mstar + z*sqrt(vLatent(:));
            pHi = ClassificationGP.normcdfSafe(mstarHi ./ denom);
        end
    end

    function L = cholSafe(X)
        % Stable Cholesky decomposition

        % Cholesky decomposition
        [L,p] = chol(X,'Lower');

        % Force positive definite
        if p
            % Force symmetry
            I = speye(size(X,1),size(X,2));
            X = 0.5.*(X+X') + 1e-12.*I;

            % Compute the symmetric polar factor of X.
            %[U,Sigma,V] = svd(X);
            %X = 0.5.*(U*Sigma*U'+V*Sigma*V');

            % Cholesky decomposition
            [L,p] = chol(X,'Lower');

            % Add adaptive jitter for final step
            max_jitter = ClassificationGP.MAX_JITTER;
            current_jitter = ClassificationGP.MIN_JITTER;
            while p && (current_jitter <= max_jitter)
                [L,p] = chol(X + current_jitter.*I, 'lower');
                current_jitter = 10*max(current_jitter,1e-10);
            end
        end

        % Return NaN when stil failing
        if p
            warning('ClassificationGP:CholeskyFailed', ...
                'Matrix is ill-conditioned.');
            L = NaN.*I;
        end
    end

    function ratio = millsRatioSafe(z)
        % Robust computation of ratio = N'(z) / N(z) = phi(z) / Phi(z)
        ratio = zeros(size(z,1),size(z,2));
        idx = (z > -8);
        if any(idx)
            ratio(idx) = ClassificationGP.normpdfSafe(z(idx))./ClassificationGP.normcdfSafe(z(idx));
        end
        if any(~idx)
            %ratio(~idx) = -z(~idx); % Simple Approx
            ratio(~idx) = (sqrt(z(~idx).^2 + 2) - z(~idx))/2; % Similar to steep sigmoid
        end
    end

    function p = normcdfSafe(x)
        % Normal CDF via erfc to avoid toolbox dependencies.
        % Safe for large inputs.
        p = 0.5 * erfc(-x./sqrt(2));
        p = min(max(p, eps), 1-eps);
    end

    function phi = normpdfSafe(x)
        % Normal PDF to avoid toolbox dependencies.
        phi = exp(-0.5.*x.^2)./sqrt(2.*pi);
    end

    function x = norminvSafe(p)
        % Normal inverse CDF via erfcinv to avoid toolbox dependencies.
        p = min(max(p, eps), 1-eps);
        x = -sqrt(2) * erfcinv(2*p);
        if ~isfinite(x)
            if p < 0.5, x = -8.5; else, x = 8.5; end
        end
    end

    function p = sigmoidSafe(f)
        % Sigmoid
        f_pos = f >= 0;
        p = zeros(size(f));
        p(f_pos) = 1./(1 + exp(-f(f_pos)));
        p(~f_pos) = exp(f(~f_pos))./(1 + exp(f(~f_pos)));
    end

    function X = peekN(Xin)
        % Utility: extract numeric matrix for size inference.
        if istable(Xin)
            X = table2array(Xin);
        else
            X = Xin;
        end
    end
end
end