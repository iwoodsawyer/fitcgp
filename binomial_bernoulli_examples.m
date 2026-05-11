%% Binomial and Bernoulli Classification Examples from Literature
% This script demonstrates practical applications of Gaussian Process Classification
% using the ClassificationGP class with binomial and Bernoulli responses

warning('off','ClassificationGP:MaxIterReached')

%% Example 1: DISEASE PRESENCE/ABSENCE (Bernoulli) - Logistic Regression Extension
% Binary classification: Disease Present (1) or Absent (0)
% Reference: Rasmussen & Williams (2006), Chapter 3: Binary Classification

fprintf('\n========================================\n');
fprintf('Example 1: Disease Presence Classification (Bernoulli)\n');
fprintf('========================================\n\n');

rng(123);
n = 200;

age = 20 + 60 * rand(n, 1);
age_norm = (age - 50) / 20;

bmi = 18 + 22 * rand(n, 1);
bmi_norm = (bmi - 28) / 7;

X_disease = [age_norm, bmi_norm];

f_disease = -1 + 0.8*age_norm + 0.6*bmi_norm;
p_disease = 1 ./ (1 + exp(-f_disease));

Y_disease_binary = (rand(n, 1) < p_disease);

fprintf('Fitting Gaussian Process Classification (Bernoulli)...\n');
mdl_disease = fitcgp(X_disease, Y_disease_binary, ...
    'Inference', 'Logit', ...
    'BasisFunction', 'linear', ...
    'Standardize', true, ...
    'Verbose', 0);

fprintf('Model Results:\n');
fprintf('  Number of observations: %d\n', mdl_disease.NumObservations);
fprintf('  Inference method: %s\n', mdl_disease.Inference);
fprintf('  Log-Likelihood: %.4f\n', mdl_disease.LogLikelihood);
fprintf('  Basis coefficients (Beta): '); fprintf('%.4f ', mdl_disease.Beta); fprintf('\n\n');

X_test_disease = [-0.5 -0.5; 0 0; 0.5 0.5];
[labels_disease, scores_disease, ci_disease] = mdl_disease.predict(X_test_disease);

fprintf('Predictions for different age/BMI combinations:\n');
fprintf('Age (norm) | BMI (norm) | P(Disease) | P(No Disease) | 95%% CI Lower | 95%% CI Upper\n');
fprintf('-----------+----------+----------+-------------+-------------+-------------\n');
for i = 1:3
    fprintf('%10.2f | %10.2f | %10.4f | %13.4f | %13.4f | %13.4f\n', ...
        X_test_disease(i,1), X_test_disease(i,2), ...
        scores_disease(i,2), scores_disease(i,1), ...
        ci_disease(i,1), ci_disease(i,2));
end
fprintf('\n');

loss_class = mdl_disease.loss(X_disease, Y_disease_binary, 'LossFun', 'classiferror');
loss_nll = mdl_disease.loss(X_disease, Y_disease_binary, 'LossFun', 'negloglikelihood');
loss_brier = mdl_disease.loss(X_disease, Y_disease_binary, 'LossFun', 'brier');

fprintf('Loss Metrics (on training data):\n');
fprintf('  Classification Error: %.4f\n', loss_class);
fprintf('  Neg. Log-Likelihood: %.4f\n', loss_nll);
fprintf('  Brier Score: %.4f\n\n', loss_brier);

%% Example 2: CREDIT DEFAULT (Binomial) - Multiple Trial Structure
% Reference: LendingClub, Kreditech datasets

fprintf('========================================\n');
fprintf('Example 2: Credit Default (Binomial Response)\n');
fprintf('========================================\n\n');

rng(456);
n_credit = 150;

credit_score = 300 + 550 * rand(n_credit, 1);
credit_norm = (credit_score - 650) / 150;

dti = 0.5 * rand(n_credit, 1);
dti_norm = (dti - 0.25) / 0.1;

X_credit = [credit_norm, dti_norm];

f_credit = -0.5 - 0.7*credit_norm + 0.5*dti_norm;
p_credit = 1 ./ (1 + exp(-f_credit));

n_trials = 20;
n_defaults = binornd(n_trials, p_credit);
Y_credit_binomial = [n_defaults, ones(n_credit, 1) * n_trials];

fprintf('Fitting Gaussian Process Classification (Binomial)...\n');
mdl_credit = fitcgp(X_credit, Y_credit_binomial, ...
    'Inference', 'Probit', ...
    'BasisFunction', 'linear', ...
    'Standardize', true, ...
    'Verbose', 0);

fprintf('Model Results:\n');
fprintf('  Number of observations: %d (each with %d trials)\n', mdl_credit.NumObservations, n_trials);
fprintf('  Inference method: %s\n', mdl_credit.Inference);
fprintf('  Log-Likelihood: %.4f\n', mdl_credit.LogLikelihood);
fprintf('  Basis coefficients (Beta): '); fprintf('%.4f ', mdl_credit.Beta); fprintf('\n\n');

X_test_credit = [-1 -1; 0 0; 1 1];
[labels_credit, scores_credit, ci_credit] = mdl_credit.predict(X_test_credit);

fprintf('Predicted default probabilities:\n');
fprintf('Credit (norm) | DTI (norm) | P(Default) | 95%% CI\n');
fprintf('-----------+----------+----------+----------\n');
for i = 1:3
    fprintf('%13.2f | %10.2f | %10.4f | [%.4f, %.4f]\n', ...
        X_test_credit(i,1), X_test_credit(i,2), ...
        scores_credit(i,2), ci_credit(i,1), ci_credit(i,2));
end
fprintf('\n');

loss_class_c = mdl_credit.loss(X_credit, Y_credit_binomial, 'LossFun', 'classiferror');
loss_nll_c = mdl_credit.loss(X_credit, Y_credit_binomial, 'LossFun', 'negloglikelihood');

fprintf('Loss Metrics:\n');
fprintf('  Classification Error: %.4f\n', loss_class_c);
fprintf('  Neg. Log-Likelihood: %.4f\n\n', loss_nll_c);

%% Example 3: EMAIL SPAM DETECTION (Bernoulli)
% Binary classification: Spam (1) or Not Spam (0)
% Reference: UCI Machine Learning Repository - Spambase dataset

fprintf('========================================\n');
fprintf('Example 3: Email Spam Detection (Bernoulli)\n');
fprintf('========================================\n\n');

rng(789);
n_spam = 250;

freq_free = 2 * rand(n_spam, 1);
freq_free_norm = (freq_free - 1) / 0.8;

freq_dollar = 2 * rand(n_spam, 1);
freq_dollar_norm = (freq_dollar - 1) / 0.8;

avg_word_len = 4 + 3 * rand(n_spam, 1);
avg_word_norm = (avg_word_len - 5.5) / 1.2;

X_spam = [freq_free_norm, freq_dollar_norm, avg_word_norm];

f_spam = -1 + 0.9*freq_free_norm + 0.8*freq_dollar_norm - 0.4*avg_word_norm;
p_spam = 1 ./ (1 + exp(-f_spam));

Y_spam = (rand(n_spam, 1) < p_spam);

fprintf('Fitting Gaussian Process Classification (Spam Detection)...\n');
mdl_spam = fitcgp(X_spam, Y_spam, ...
    'Inference', 'Logit', ...
    'BasisFunction', 'linear', ...
    'KernelFunction', 'squaredexponential', ...
    'Standardize', true, ...
    'Verbose', 0);

fprintf('Model Results:\n');
fprintf('  Number of observations: %d\n', mdl_spam.NumObservations);
fprintf('  Inference: %s\n', mdl_spam.Inference);
fprintf('  Kernel: %s\n', mdl_spam.KernelFunction);
fprintf('  Log-Likelihood: %.4f\n', mdl_spam.LogLikelihood);
fprintf('  Basis coefficients: '); fprintf('%.4f ', mdl_spam.Beta); fprintf('\n\n');

X_test_spam = [
    1  1  -1;
    -1 -1  1;
    0  0   0
];

[labels_spam, scores_spam] = mdl_spam.predict(X_test_spam);

fprintf('Email Classification Predictions:\n');
fprintf('Email Type | Freq(free) | Freq($) | Word_Len | P(Spam) | Predicted Label\n');
fprintf('-----------+----------+----------+---------+---------+----------------\n');
email_types = {'Likely spam', 'Likely ham', 'Uncertain'};
for i = 1:3
    fprintf('%-10s | %10.2f | %9.2f | %8.2f | %7.4f | %s\n', ...
        email_types{i}, X_test_spam(i,1), X_test_spam(i,2), X_test_spam(i,3), ...
        scores_spam(i,2), char(labels_spam(i)));
end
fprintf('\n');

%% Example 4: A/B TESTING - CONVERSION RATE (Binomial)

fprintf('========================================\n');
fprintf('Example 4: A/B Testing - Conversion Rate (Binomial)\n');
fprintf('========================================\n\n');

rng(999);
n_variants = 100;

traffic_log = log(100 + 900*rand(n_variants, 1));
traffic_norm = (traffic_log - mean(traffic_log)) / std(traffic_log);

button_color = randi([0, 1], n_variants, 1) - 0.5;

X_ab = [traffic_norm, button_color];

f_ab = -2 + 0.4*traffic_norm + 0.3*button_color;
p_ab = 1 ./ (1 + exp(-f_ab));

n_users_per_variant = 100;
conversions = binornd(n_users_per_variant, p_ab);
Y_ab = [conversions, ones(n_variants, 1) * n_users_per_variant];

fprintf('Fitting Gaussian Process Classification (A/B Test)...\n');
mdl_ab = fitcgp(X_ab, Y_ab, ...
    'Inference', 'Probit', ...
    'BasisFunction', 'constant', ...
    'Standardize', true, ...
    'Verbose', 0);

fprintf('Model Results:\n');
fprintf('  Number of A/B variants: %d\n', mdl_ab.NumObservations);
fprintf('  Users per variant: %d\n', n_users_per_variant);
fprintf('  Inference: %s\n', mdl_ab.Inference);
fprintf('  Log-Likelihood: %.4f\n\n', mdl_ab.LogLikelihood);

X_test_ab = [-1 -0.5; 0 0; 1 0.5];
[~, scores_ab] = mdl_ab.predict(X_test_ab);

fprintf('Predicted Conversion Rates:\n');
fprintf('Traffic (norm) | Button | P(Conversion) | Expected Conversions (of 100)\n');
fprintf('-----------+----------+-------+---+----------\n');
buttons = {'Red', 'Green'};
for i = 1:3
    traffic_val = X_test_ab(i,1);
    button_idx = X_test_ab(i,2);
    if button_idx < 0
        button_str = 'Red';
    else
        button_str = 'Green';
    end
    expected_conv = scores_ab(i,2) * 100;
    fprintf('%13.2f | %6s | %13.4f | %27.1f\n', ...
        traffic_val, button_str, scores_ab(i,2), expected_conv);
end
fprintf('\n');

%% Example 5: INFERENCE COMPARISON (Logit vs Probit vs EP)

fprintf('========================================\n');
fprintf('Example 5: Inference Comparison (Logit vs Probit vs EP)\n');
fprintf('========================================\n\n');

rng(111);
n_inf = 150;

X_inf = [-1 + 2*rand(n_inf, 1), -1 + 2*rand(n_inf, 1)];
f_inf = 0.5 + 0.7*X_inf(:,1) - 0.5*X_inf(:,2);
Y_inf = (rand(n_inf, 1) < 1./(1+exp(-f_inf)));

fprintf('Fitting with different inference methods...\n\n');

mdl_logit = fitcgp(X_inf, Y_inf, ...
    'Inference', 'Logit', ...
    'BasisFunction', 'linear', ...
    'Standardize', true, ...
    'ProbitScaling', 'none', ...
    'Verbose', 0);

mdl_probit = fitcgp(X_inf, Y_inf, ...
    'Inference', 'Probit', ...
    'BasisFunction', 'linear', ...
    'Standardize', true, ...
    'ProbitScaling', 'none', ...
    'Verbose', 0);

mdl_ep = fitcgp(X_inf, Y_inf, ...
    'Inference', 'EP', ...
    'BasisFunction', 'linear', ...
    'Standardize', true, ...
    'Verbose', 0);

fprintf('Inference Methods Comparison:\n');
fprintf('Method    | Log-Likelihood | Basis Coefficients\n');
fprintf('-----------+----------------+------------------\n');
fprintf('Logit (Laplace) | %14.4f | '); fprintf('%.4f ', mdl_logit.Beta); fprintf('\n');
fprintf('Probit (Laplace)| %14.4f | '); fprintf('%.4f ', mdl_probit.Beta); fprintf('\n');
fprintf('EP              | %14.4f | '); fprintf('%.4f ', mdl_ep.Beta); fprintf('\n\n');

X_test_inf = [-0.5 -0.5; 0 0; 0.5 0.5];
[~, scores_logit] = mdl_logit.predict(X_test_inf);
[~, scores_probit] = mdl_probit.predict(X_test_inf);
[~, scores_ep] = mdl_ep.predict(X_test_inf);

fprintf('Predicted Probabilities Comparison:\n');
fprintf('Test Point | Logit | Probit | EP    | Logit-Probit Diff\n');
fprintf('-----------+-------+-------+-------+------------------\n');
for i = 1:3
    diff = abs(scores_logit(i,2) - scores_probit(i,2));
    fprintf('[%5.2f,%5.2f] | %.4f | %.4f | %.4f | %17.6f\n', ...
        X_test_inf(i,1), X_test_inf(i,2), ...
        scores_logit(i,2), scores_probit(i,2), scores_ep(i,2), diff);
end
fprintf('\n');

%% Example 6: MODEL SELECTION

fprintf('========================================\n');
fprintf('Example 6: Model Selection with Information Criteria\n');
fprintf('========================================\n\n');

rng(222);
n_select = 200;

X_select = [-1 + 2*rand(n_select, 1), -1 + 2*rand(n_select, 1)];
f_select = 0.3*X_select(:,1) + 0.4*X_select(:,2);
Y_select = (rand(n_select, 1) < 1./(1+exp(-f_select)));

fprintf('Fitting models with different configurations...\n\n');

configs = {
    struct('name', 'Const + SE', 'basis', 'constant', 'kernel', 'squaredexponential'),
    struct('name', 'Linear + SE', 'basis', 'linear', 'kernel', 'squaredexponential'),
    struct('name', 'Const + Matern52', 'basis', 'constant', 'kernel', 'matern52'),
    struct('name', 'Linear + Matern52', 'basis', 'linear', 'kernel', 'matern52')
};

results_select = [];

for i = 1:numel(configs)
    cfg = configs{i};
    mdl = fitcgp(X_select, Y_select, ...
        'BasisFunction', cfg.basis, ...
        'KernelFunction', cfg.kernel, ...
        'Standardize', true, ...
        'Verbose', 0);
    
    aic = mdl.criterion('CriterionFun', 'AIC');
    bic = mdl.criterion('CriterionFun', 'BIC');
    loss = mdl.resubLoss('LossFun', 'classiferror');
    
    results_select = [results_select; aic, bic, loss, numel(mdl.Beta)];
    
    fprintf('%-20s | AIC: %10.4f | BIC: %10.4f | ClassError: %.4f | Beta: %d\n', ...
        cfg.name, aic, bic, loss, numel(mdl.Beta));
end

fprintf('\n');
[~, best_aic] = min(results_select(:,1));
[~, best_bic] = min(results_select(:,2));
fprintf('Best AIC: %s\n', configs{best_aic}.name);
fprintf('Best BIC: %s\n\n', configs{best_bic}.name);

%% Summary
fprintf('========================================\n');
fprintf('Summary: Binomial & Bernoulli Classification Examples\n');
fprintf('========================================\n\n');
fprintf('Classification Problems Demonstrated:\n');
fprintf('  1. Disease Classification: Bernoulli (binary 0/1)\n');
fprintf('  2. Credit Default: Binomial (count out of N trials)\n');
fprintf('  3. Spam Detection: Bernoulli (binary 0/1)\n');
fprintf('  4. A/B Testing: Binomial (conversions/users)\n');
fprintf('  5. Inference Methods: Logit vs Probit vs EP\n');
fprintf('  6. Model Selection: AIC/BIC comparison\n\n');
fprintf('Key Features:\n');
fprintf('  - Gaussian Process prior for flexible decision boundaries\n');
fprintf('  - Logit and Probit link functions\n');
fprintf('  - Laplace approximation and Expectation Propagation\n');
fprintf('  - Confidence intervals for predicted probabilities\n');
fprintf('  - Multiple loss functions (classification error, Brier, NLL)\n');
fprintf('  - Model comparison with AIC/BIC\n');