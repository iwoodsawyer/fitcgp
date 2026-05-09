classdef test_fitcgp < matlab.unittest.TestCase
    % TEST_FITCGP Consolidated Qualification Tests for ClassificationGP (fitcgp)
    % Combines scenarios from test.m, test_qual_gemini, test_qual_gpt, and test_qual.

    properties
        % 3D Synthetic Data (from test_qual suites)
        X_bin
        Y_cat
        Y_logical
        Y_numeric
        Y_binom
        Tbl
        
        % 1D Real-ish Data (from test.m)
        X_1d
        N_1d
        Y_1d
        P_1d
    end

    methods (TestMethodSetup)
        function createData(testCase)
            rng(42);
            
            % --- 1. Synthetic 3D Data ---
            n = 100;
            x1 = randn(n,1);
            x2 = randn(n,1);
            x3 = randn(n,1);
            testCase.X_bin = [x1, x2, x3];

            eta  = 1.5*x1 - 0.8*x2 + 0.4*x3;
            p    = 1 ./ (1 + exp(-eta));
            yLog = rand(n,1) < p;

            testCase.Y_logical = yLog;
            testCase.Y_numeric = double(yLog);
            testCase.Y_cat     = categorical(yLog, [false true], {'ClassA','ClassB'});

            trials    = randi([5 15], n, 1);
            successes = binornd(trials, p);
            testCase.Y_binom = [successes, trials];

            testCase.Tbl = table(x1, x2, x3, testCase.Y_cat, ...
                'VariableNames', {'x1','x2','x3','Response'});
                
            % --- 2. 1D Binomial/Proportion Data (from test.m) ---
            testCase.X_1d = [2100 2300 2500 2700 2900 3100 3300 3500 3700 3900 4100 4300]';
            testCase.N_1d = [48 42 31 34 31 21 23 23 21 16 17 21]';
            testCase.Y_1d = [1 2 0 3 8 8 14 17 19 15 17 21]';
            testCase.P_1d = testCase.Y_1d ./ testCase.N_1d;
        end
    end

    % =========================================================
    methods (Test)

        % ---------------------------------------------------------
        % 1. ORIGINAL SCRIPT WORKFLOWS (from test.m)
        % ---------------------------------------------------------
        function test1DScriptWorkflows(testCase)
            x = testCase.X_1d;
            n = testCase.N_1d;
            y = testCase.Y_1d;
            p = testCase.P_1d;

            % Logit, Probit, EP Inference with Proportions
            Mdl = fitcgp(x, p, 'Inference', 'Logit', 'BasisFunction', 'Constant', 'Standardize', true, 'Verbose', 0);
            [~, scores] = predict(Mdl, x);
            testCase.verifySize(scores, [12 2]);

            Mdl = fitcgp(x, p, 'Inference', 'Probit', 'BasisFunction', 'Constant', 'Standardize', true, 'Verbose', 0);
            predict(Mdl, x);

            Mdl = fitcgp(x, p, 'Inference', 'EP', 'BasisFunction', 'Constant', 'Standardize', true, 'Verbose', 0);
            predict(Mdl, x);

            % Loss and Criterion Workflows
            Mdl = fitcgp(x, p, 'BasisFunction', 'Constant', 'Standardize', true, 'Verbose', 0);
            predict(Mdl, x);
            
            lossFuns = {'Quadratic', 'Hinge', 'Negloglikelihood', 'Brier', 'CRPS', 'DSS'};
            for i = 1:numel(lossFuns)
                L = loss(Mdl, x, p, 'LossFun', lossFuns{i});
                testCase.verifyTrue(isfinite(L));
            end
            
            critFuns = {'AIC', 'AICc', 'BIC', 'CAIC', 'GCV', 'LOOCV', 'WAIC'};
            for i = 1:numel(critFuns)
                C = criterion(Mdl, 'CriterionFun', critFuns{i});
                testCase.verifyTrue(isfinite(C));
            end

            % Binomial Input [y n] with Linear Basis
            Mdl = fitcgp(x, [y n], 'BasisFunction', 'Linear', 'Standardize', true, 'Verbose', 0);
            predict(Mdl, x);
            L = loss(Mdl, x, [y n], 'LossFun', 'Negloglikelihood');
            testCase.verifyTrue(isfinite(L));

            % Bayesopt workflow (restricted evaluations for test speed)
            opts = struct('MaxObjectiveEvaluations', 2, 'Verbose', 0);
            Mdl = compact(fitcgp(x, p, 'BasisFunction', 'Constant', ...
                'OptimizeHyperparameters', 'auto', ...
                'HyperparameterOptimizationOptions', opts, ...
                'Standardize', true, 'KFold', 2, 'Verbose', 0));
            [~, score, ci] = predict(Mdl, x);
            testCase.verifySize(score, [12 2]);
            testCase.verifySize(ci, [12 2]);
        end

        % ---------------------------------------------------------
        % 2. OUTPUT SHAPE & SANITY
        % ---------------------------------------------------------
        function testOutputShapeAndBounds(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Verbose', 0);

            Xte = testCase.X_bin(1:15,:);
            [label, score, ci] = mdl.predict(Xte);

            testCase.verifySize(label,  [15 1]);
            testCase.verifySize(score,  [15 2]);
            testCase.verifySize(ci,     [15 2]);

            testCase.verifyTrue(all(isfinite(score(:))));
            testCase.verifyTrue(all(score(:) >= 0 & score(:) <= 1));
            testCase.verifyLessThan(max(abs(sum(score,2) - 1)), 1e-8);

            testCase.verifyTrue(all(ci(:,1) <= score(:,2)));
            testCase.verifyTrue(all(ci(:,2) >= score(:,2)));
        end

        % ---------------------------------------------------------
        % 3. RESPONSE TYPE EQUIVALENCE
        % ---------------------------------------------------------
        function testResponseTypesGiveSimilarScores(testCase)
            Xte   = testCase.X_bin(1:20,:);
            opts  = {'Verbose',0,'FitMethod','none'};
            s_cat = fitcgp(testCase.X_bin, testCase.Y_cat,     opts{:}).predict(Xte);
            s_log = fitcgp(testCase.X_bin, testCase.Y_logical, opts{:}).predict(Xte);
            s_num = fitcgp(testCase.X_bin, testCase.Y_numeric, opts{:}).predict(Xte);

            testCase.verifyTrue(all((s_cat=='ClassB') == (s_log=='1')));
            testCase.verifyTrue(all((s_cat=='ClassB') == (s_num=='1')));
        end

        % ---------------------------------------------------------
        % 4. TABLE INPUT AND RESPONSE NAME
        % ---------------------------------------------------------
        function testTableInput(testCase)
            mdl = fitcgp(testCase.Tbl, 'Response', 'Verbose', 0);

            testCase.verifyEqual(string(mdl.ResponseName), "Response");
            testCase.verifyEqual(numel(mdl.PredictorNames), 3);

            [~, score] = mdl.predict(testCase.Tbl(1:10, 1:3));
            testCase.verifySize(score, [10 2]);
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        % ---------------------------------------------------------
        % 5. INFERENCE METHODS
        % ---------------------------------------------------------
        function testInferenceLogit(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Inference', 'Logit', 'Verbose', 0);
            testCase.verifyEqual(lower(mdl.Inference), 'laplace');
            [~,score] = mdl.predict(testCase.X_bin(1:10,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testInferenceProbit(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Inference', 'Probit', 'Verbose', 0);
            testCase.verifyEqual(lower(mdl.Inference), 'laplace');
            [~,score] = mdl.predict(testCase.X_bin(1:10,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testInferenceEP(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Inference', 'EP', 'Verbose', 0);
            testCase.verifyEqual(lower(mdl.Inference), 'ep');
            [~,score] = mdl.predict(testCase.X_bin(1:10,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testInferenceEPWithWeights(testCase)
            w = 0.5 + rand(size(testCase.X_bin,1),1);
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, ...
                'Inference', 'EP', ...
                'Weights', w, ...
                'Verbose', 0);
            testCase.verifyEqual(lower(mdl.Inference), 'ep');
            [~, score] = mdl.predict(testCase.X_bin(1:10,:));
            testCase.verifyTrue(all(isfinite(score(:))));
            testCase.verifyTrue(all(score(:) >= 0 & score(:) <= 1));
        end

        function testInferenceLogitVsProbitDiffer(testCase)
            opts = {'Verbose',0};
            Xte  = testCase.X_bin(1:20,:);
            [~, s_logit]  = fitcgp(testCase.X_bin, testCase.Y_cat, 'Inference','Logit',  opts{:}).predict(Xte);
            [~, s_probit] = fitcgp(testCase.X_bin, testCase.Y_cat, 'Inference','Probit', opts{:}).predict(Xte);
            testCase.verifyGreaterThan(max(abs(s_logit(:,2) - s_probit(:,2))), 1e-6);
        end

        % ---------------------------------------------------------
        % 6. BINOMIAL INPUT
        % ---------------------------------------------------------
        function testBinomialInputProbit(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_binom, 'Inference', 'Probit', 'Verbose', 0);
            [~, score] = mdl.predict(testCase.X_bin(1:10,:));
            testCase.verifySize(score, [10 2]);
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testBinomialInputLogit(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_binom, 'Inference', 'Logit', 'Verbose', 0);
            [~, score] = mdl.predict(testCase.X_bin(1:10,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        % ---------------------------------------------------------
        % 7. KERNEL FUNCTIONS
        % ---------------------------------------------------------
        function testKernelSquaredExponential(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'KernelFunction', 'squaredexponential', 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:5,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testKernelExponential(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'KernelFunction', 'exponential', 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:5,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testKernelMatern32(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'KernelFunction', 'matern32', 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:5,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testKernelMatern52(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'KernelFunction', 'matern52', 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:5,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testKernelRationalQuadratic(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'KernelFunction', 'rationalquadratic', 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:5,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testKernelARDSquaredExponential(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'KernelFunction', 'ardsquaredexponential', 'Verbose', 0);
            testCase.verifyEqual(numel(mdl.KernelInformation.KernelParameters), size(testCase.X_bin,2)+1);
            [~,score] = mdl.predict(testCase.X_bin(1:5,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testKernelARDMatern52(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'KernelFunction', 'ardmatern52', 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:5,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testKernelFunctionHandle(testCase)
            sqexp = @(X1,X2,theta) theta(2)^2 .* exp(-0.5 .* pdist2(X1./theta(1), X2./theta(1)).^2);
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, ...
                'KernelFunction', sqexp, ...
                'KernelParameters', [1;1], 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:5,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        % ---------------------------------------------------------
        % 8. BASIS FUNCTIONS
        % ---------------------------------------------------------
        function testBasisNone(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'BasisFunction', 'none', 'Verbose', 0);
            testCase.verifyEqual(numel(mdl.Beta), 0);
        end

        function testBasisConstant(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'BasisFunction', 'constant', 'Verbose', 0);
            testCase.verifyEqual(numel(mdl.Beta), 1);
        end

        function testBasisLinear(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'BasisFunction', 'linear', 'Verbose', 0);
            testCase.verifyEqual(numel(mdl.Beta), size(testCase.X_bin,2)+1);
        end

        function testBasisPureQuadratic(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'BasisFunction', 'purequadratic', 'Verbose', 0);
            testCase.verifyEqual(numel(mdl.Beta), 2*size(testCase.X_bin,2)+1);
        end

        function testBasisFunctionHandle(testCase)
            customH = @(X) [ones(size(X,1),1), X, X.^2];
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, ...
                'BasisFunction', customH, ...
                'FitMethod', 'none', 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:5,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        % ---------------------------------------------------------
        % 9. LAMBDA
        % ---------------------------------------------------------
        function testLambdaZeroAllowed(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Lambda', 0, 'ConstantLambda', true, 'Verbose', 0);
            testCase.verifyEqual(mdl.Lambda, 0);
        end

        function testLambdaNonZero(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Lambda', 1e-4, 'ConstantLambda', true, 'Verbose', 0);
            testCase.verifyEqual(mdl.Lambda, 1e-4);
            [~,score] = mdl.predict(testCase.X_bin(1:5,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        % ---------------------------------------------------------
        % 10. STANDARDIZE
        % ---------------------------------------------------------
        function testStandardizeFalse(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Standardize', false, 'FitMethod', 'none', 'Verbose', 0);
            testCase.verifyEqual(mdl.PredictorLocation, zeros(1,3));
            testCase.verifyEqual(mdl.PredictorScale,    ones(1,3));
        end

        function testStandardizeTrue(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Standardize', true, 'FitMethod', 'none', 'Verbose', 0);
            testCase.verifyNotEqual(mdl.PredictorLocation, zeros(1,3));
            testCase.verifyNotEqual(mdl.PredictorScale,    ones(1,3));
        end

        % ---------------------------------------------------------
        % 11. ACTIVE SET METHODS
        % ---------------------------------------------------------
        function testActiveSetRandom(testCase)
            m = 40;
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'ActiveSetMethod', 'random', 'ActiveSetSize', m, 'Verbose', 0);
            testCase.verifyEqual(sum(mdl.IsActiveSetVector), m);
        end

        function testActiveSetFirst(testCase)
            m = 40;
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'ActiveSetMethod', 'first', 'ActiveSetSize', m, 'Verbose', 0);
            testCase.verifyEqual(sum(mdl.IsActiveSetVector), m);
            testCase.verifyTrue(all(find(mdl.IsActiveSetVector) == (1:m)'));
        end

        function testActiveSetLast(testCase)
            m = 40;
            n = size(testCase.X_bin,1);
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'ActiveSetMethod', 'last', 'ActiveSetSize', m, 'Verbose', 0);
            testCase.verifyEqual(sum(mdl.IsActiveSetVector), m);
            testCase.verifyTrue(all(find(mdl.IsActiveSetVector) == (n-m+1:n)'));
        end

        function testActiveSetQR(testCase)
            m = 40;
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'ActiveSetMethod', 'qr', 'ActiveSetSize', m, 'Verbose', 0);
            testCase.verifyEqual(sum(mdl.IsActiveSetVector), m);
        end

        function testExplicitActiveSetVectorAndIndices(testCase)
            % Explicit Logical Vector
            isActive = false(size(testCase.X_bin,1),1);
            isActive(1:20) = true;
            mdlLogical = fitcgp(testCase.X_bin, testCase.Y_cat, 'ActiveSet', isActive, 'Verbose', 0);

            % Explicit Numeric Indices
            mdlNumeric = fitcgp(testCase.X_bin, testCase.Y_cat, 'ActiveSet', (1:20)', 'Verbose', 0);

            [~, scoresLogical] = mdlLogical.predict(testCase.X_bin(1:4,:));
            [~, scoresNumeric] = mdlNumeric.predict(testCase.X_bin(1:4,:));

            testCase.verifyEqual(sum(mdlLogical.IsActiveSetVector), 20);
            testCase.verifyEqual(sum(mdlNumeric.IsActiveSetVector), 20);
            testCase.verifyEqual(size(scoresLogical), [4 2]);
            testCase.verifyEqual(size(scoresNumeric), [4 2]);
        end

        % ---------------------------------------------------------
        % 12. PROBIT SCALING
        % ---------------------------------------------------------
        function testProbitScalingNone(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Inference', 'Probit', 'ProbitScaling', 'none', 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:10,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testProbitScalingSlope(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Inference', 'Probit', 'ProbitScaling', 'slope', 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:10,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        function testProbitScalingMinimax(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Inference', 'Probit', 'ProbitScaling', 'minimax', 'Verbose', 0);
            [~,score] = mdl.predict(testCase.X_bin(1:10,:));
            testCase.verifyTrue(all(isfinite(score(:))));
        end

        % ---------------------------------------------------------
        % 13. FIT METHOD
        % ---------------------------------------------------------
        function testFitMethodNone(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'FitMethod', 'none', 'Verbose', 0);
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testFitMethodExact(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'FitMethod', 'exact', 'Verbose', 0);
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        % ---------------------------------------------------------
        % 14. CONFIDENCE INTERVAL
        % ---------------------------------------------------------
        function testCIDefaultAlpha(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Verbose', 0);
            Xte = testCase.X_bin(1:10,:);
            [~, score, ci] = mdl.predict(Xte);
            testCase.verifyTrue(all(ci(:,1) <= score(:,2) + 1e-10));
            testCase.verifyTrue(all(ci(:,2) >= score(:,2) - 1e-10));
        end

        % ---------------------------------------------------------
        % 15. LOSS FUNCTIONS
        % ---------------------------------------------------------
        function testLossFunctions(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Verbose', 0);
            lossFuns = {'classiferror', 'hinge', 'quadratic', 'negloglikelihood', 'brier', 'crps', 'dss'};
            
            for i = 1:length(lossFuns)
                L = mdl.loss(testCase.X_bin, testCase.Y_cat, 'LossFun', lossFuns{i});
                testCase.verifyTrue(isfinite(L));
            end
        end

        function testResubLoss(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Verbose', 0);
            L = mdl.resubLoss();
            testCase.verifyTrue(isfinite(L));
        end

        % ---------------------------------------------------------
        % 16. CRITERION FUNCTIONS
        % ---------------------------------------------------------
        function testCriterionFunctions(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Verbose', 0);
            critFuns = {'aic', 'aicc', 'bic', 'caic', 'gcv', 'loocv', 'waic'};
            
            for i = 1:length(critFuns)
                C = mdl.criterion('CriterionFun', critFuns{i});
                testCase.verifyTrue(isfinite(C));
            end
        end

        function testCriterionAIClessThanBIC(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Verbose', 0);
            aic = mdl.criterion('CriterionFun', 'aic');
            bic = mdl.criterion('CriterionFun', 'bic');
            k   = numel(mdl.KernelInformation.KernelParameters) + numel(mdl.Beta);
            n   = mdl.NumObservations;
            testCase.verifyEqual(bic - aic, k*(log(n) - 2), 'AbsTol', 1e-8);
        end

        % ---------------------------------------------------------
        % 17. COMPACT
        % ---------------------------------------------------------
        function testCompactRemovesData(testCase)
            mdl  = fitcgp(testCase.X_bin, testCase.Y_cat, 'Verbose', 0);
            cMdl = mdl.compact();
            testCase.verifyEmpty(cMdl.X);
            testCase.verifyEmpty(cMdl.Y);
            testCase.verifyEmpty(cMdl.W);
            testCase.verifyEqual(cMdl.NumObservations, 0);
        end

        function testCompactPreservesPredict(testCase)
            mdl  = fitcgp(testCase.X_bin, testCase.Y_cat, 'Verbose', 0);
            Xte  = testCase.X_bin(1:10,:);
            [~, s_full]    = mdl.predict(Xte);
            [~, s_compact] = mdl.compact().predict(Xte);
            testCase.verifyLessThan(max(abs(s_full(:,2) - s_compact(:,2))), 1e-10);
        end

        function testCompactResubErrors(testCase)
            cMdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Verbose', 0).compact();
            testCase.verifyError(@() cMdl.resubLoss(),    'ClassificationGP:Compact');
            testCase.verifyError(@() cMdl.resubPredict(), 'ClassificationGP:Compact');
            testCase.verifyError(@() cMdl.criterion(),    'ClassificationGP:Compact');
        end

        % ---------------------------------------------------------
        % 18. RESUBPREDICT
        % ---------------------------------------------------------
        function testResubPredictMatchesPredict(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'Verbose', 0);
            [~, s_resub] = mdl.resubPredict();
            [~, s_pred]  = mdl.predict(testCase.X_bin);
            testCase.verifyLessThan(max(abs(s_resub(:,2) - s_pred(:,2))), 1e-10);
        end

        % ---------------------------------------------------------
        % 19. PREDICTOR AND RESPONSE NAME PASSTHROUGH
        % ---------------------------------------------------------
        function testPredictorNames(testCase)
            names = {'feat1','feat2','feat3'};
            mdl   = fitcgp(testCase.X_bin, testCase.Y_cat, 'PredictorNames', names, 'Verbose', 0);
            testCase.verifyEqual(mdl.PredictorNames, string(names));
        end

        function testResponseName(testCase)
            mdl = fitcgp(testCase.X_bin, testCase.Y_cat, 'ResponseName', 'outcome', 'Verbose', 0);
            testCase.verifyEqual(string(mdl.ResponseName), "outcome");
        end

        % ---------------------------------------------------------
        % 20. INPUT VALIDATION ERRORS
        % ---------------------------------------------------------
        function testEmptyXErrors(testCase)
            testCase.verifyError(@() fitcgp([], testCase.Y_cat), 'ClassificationGP:EmptyX');
        end

        function testEmptyYErrors(testCase)
            testCase.verifyError(@() fitcgp(testCase.X_bin, []), 'ClassificationGP:EmptyY');
        end

        function testSizeMismatchErrors(testCase)
            testCase.verifyError(@() fitcgp(testCase.X_bin, testCase.Y_cat(1:10)), 'ClassificationGP:SizeMismatch');
        end

        function testNaNInXErrors(testCase)
            Xbad = testCase.X_bin;
            Xbad(1,1) = NaN;
            testCase.verifyError(@() fitcgp(Xbad, testCase.Y_cat), 'ClassificationGP:InvalidX');
        end

        function testMoreThanTwoClassesErrors(testCase)
            Ymulti = categorical(randi(3, size(testCase.X_bin,1), 1));
            testCase.verifyError(@() fitcgp(testCase.X_bin, Ymulti), 'ClassificationGP:BinaryOnly');
        end

    end
end