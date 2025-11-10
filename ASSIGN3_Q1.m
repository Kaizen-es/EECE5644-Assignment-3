%{
                    EECE5644 FALL 2025 - ASSIGNMENT 3
                                QUESTION 1
                    MLP CLASSIFICATION WITH CROSS-VALIDATION
%}

clear all, close all,

% Set random seed for reproducibility
rng(5644, 'twister');

fprintf('ASSIGNMENT 3 - QUESTION 1\n');
fprintf('MLP Classification with Cross-Validation\n\n');

% STEP 1: DESIGN 4-CLASS GMM IN 3D SPACE
% Data parameters
C = 4;          
n = 3;          
p = ones(1,C)/C;  

% Mean vectors
mu(:,1) = [-1.6; -1.6; -1.6];   
mu(:,2) = [ 1.6; -1.6;  1.6];   
mu(:,3) = [-1.6;  1.6;  1.6];   
mu(:,4) = [ 1.6;  1.6; -1.6];

% Covariance matrices 
Sigma(:,:,1) = [2.0  0.2  0.2;
                0.2  2.0  0.2;
                0.2  0.2  2.0];

Sigma(:,:,2) = [2.5  0.0  0.8;
                0.0  1.8  0.0;
                0.8  0.0  2.5];

Sigma(:,:,3) = [1.8  0.0  0.0;
                0.0  2.5  0.8;
                0.0  0.8  2.5];

Sigma(:,:,4) = [2.5  0.3  0.3;
                0.3  2.5  0.3;
                0.3  0.3  2.5];

% Setup GMM parameters - Pattern from Professor's generateDataFromGMM.m
gmmParameters.priors = p;
gmmParameters.meanVectors = mu;
gmmParameters.covMatrices = Sigma;

fprintf('GMM Configuration:\n');
fprintf('  Classes: %d\n', C);
fprintf('  Dimensions: %d\n', n);
fprintf('  Priors: [%.2f, %.2f, %.2f, %.2f] (uniform)\n\n', p);

fprintf('DATASETS\n');

% Training set sizes
N_train = [100, 500, 1000, 5000, 10000];

% Generate training datasets - Using Professor's generateDataFromGMM.m
x_train = cell(length(N_train), 1);
labels_train = cell(length(N_train), 1);

for i = 1:length(N_train)
    [x_train{i}, labels_train{i}] = generateDataFromGMM(N_train(i), gmmParameters);
    
    N_c = zeros(1,C);
    for l = 1:C
        N_c(l) = length(find(labels_train{i}==l));
    end
    
    fprintf('  D_train^%d: %d samples (', N_train(i), N_train(i));
    for l = 1:C
        fprintf('Class %d: %d', l, N_c(l));
        if l < C
            fprintf(', ');
        end
    end
    fprintf(')\n');
end

% Generate test dataset - Using Professor's generateDataFromGMM.m
N_test = 100000;
[x_test, labels_test] = generateDataFromGMM(N_test, gmmParameters);

N_c_test = zeros(1,C);
for l = 1:C
    N_c_test(l) = length(find(labels_test==l));
end

fprintf('  D_test: %d samples (', N_test);
for l = 1:C
    fprintf('Class %d: %d', l, N_c_test(l));
    if l < C
        fprintf(', ');
    end
end
fprintf(')\n\n');

% Visualize test data - Pattern from Professor's ExpectedRiskMinimization.m (lines 43-45)
figure(1), clf,
plot3(x_test(1,labels_test==1), x_test(2,labels_test==1), x_test(3,labels_test==1), '.b'); hold on;
plot3(x_test(1,labels_test==2), x_test(2,labels_test==2), x_test(3,labels_test==2), '.r');
plot3(x_test(1,labels_test==3), x_test(2,labels_test==3), x_test(3,labels_test==3), '.g');
plot3(x_test(1,labels_test==4), x_test(2,labels_test==4), x_test(3,labels_test==4), '.c');
axis equal, grid on,
xlabel('x_1'), ylabel('x_2'), zlabel('x_3');
title('Test Data: 4-Class GMM in 3D');
legend('Class 1', 'Class 2', 'Class 3', 'Class 4');
view(45, 30);

%THEORETICALLY OPTIMAL CLASSIFIER

fprintf('THEORETICALLY OPTIMAL CLASSIFIER\n');

% Compute class-conditional likelihoods - Pattern from Professor's ERMwithClabels.m
pxgivenl = zeros(C, N_test);
for l = 1:C
    pxgivenl(l,:) = evalGaussian(x_test, mu(:,l), Sigma(:,:,l));
end

% Compute class posteriors - Pattern from Professor's ERMwithClabels.m
px = p * pxgivenl;
classPosteriors = pxgivenl .* repmat(p', 1, N_test) ./ repmat(px, C, 1);

% MAP decision rule - Pattern from Professor's ERMwithClabels.m
[~, decisions] = max(classPosteriors, [], 1);

% Calculate probability of error
P_error_optimal = sum(decisions ~= labels_test) / N_test;

fprintf('  P(error) = %.4f (%.2f%%)\n', P_error_optimal, 100*P_error_optimal);


% Compute confusion matrix - Pattern from Professor's ERMwithClabels.m
ConfusionMatrix = zeros(C,C);
for d = 1:C
    for l = 1:C
        ind_dl = find(decisions==d & labels_test==l);
        ConfusionMatrix(d,l) = length(ind_dl) / N_c_test(l);
    end
end

fprintf('Per-Class Accuracy:\n');
for l = 1:C
    fprintf('  Class %d: %.4f (%.2f%%)\n', l, ConfusionMatrix(l,l), 100*ConfusionMatrix(l,l));
end
fprintf('\n');

% Visualize classification results - Pattern adapted from discussion with Chatgpt
figure(2), clf, hold on

correct = (decisions == labels_test);
incorrect = ~correct;
plot3(x_test(1,correct), x_test(2,correct), x_test(3,correct), 'go', ...
      'MarkerSize', 4, 'DisplayName', 'Correct');
plot3(x_test(1,incorrect), x_test(2,incorrect), x_test(3,incorrect), 'rx', ...
      'MarkerSize', 4, 'DisplayName', 'Incorrect');

axis equal, grid on
xlabel('x_1'), ylabel('x_2'), zlabel('x_3');
title(sprintf('Optimal Classification Results: P(error) = %.4f', P_error_optimal));
legend('Correct','Incorrect','Location','best');
view(45, 30);

% MLP TRAINING WITH CROSS-VALIDATION
fprintf('MLP TRAINING WITH CROSS-VALIDATION\n');
% Perceptron values to try
P_values = [2, 4, 8, 16];
K = 10; % 10-fold cross-validation

% Storage for results
P_optimal = zeros(length(N_train), 1);
cv_errors = zeros(length(N_train), length(P_values));
trained_models = cell(length(N_train), 1);

% For each training set size
for i = 1:length(N_train)
    fprintf('\nTraining set size: N = %d\n', N_train(i));
    
    % Get current training data (keep as n x N format for consistency with professor's code)
    X_current = x_train{i};  % 3 x N
    Y_current = labels_train{i};  % 1 x N
    
    % Try each perceptron count
    fprintf('  P values: ');
    for j = 1:length(P_values)
        P = P_values(j);
        fprintf('%d ', P);
        
        % 10-fold cross-validation - Structure adapted from Professor's 
        % PolynomialFitCrossValidation.m (lines 34-52: K-fold loop structure and error averaging)
        cv = cvpartition(Y_current, 'KFold', K);
        fold_errors = zeros(K, 1);
        
        for k = 1:K
            % Split data - Partition approach adapted from Professor's kfoldMLP.m
            train_idx = training(cv, k);
            val_idx = test(cv, k);
            
            X_fold_train = X_current(:, train_idx);
            Y_fold_train = Y_current(train_idx);
            X_fold_val = X_current(:, val_idx);
            Y_fold_val = Y_current(val_idx);
            
            % Train MLP with P perceptrons using custom implementation
            Mdl = trainCustomMLP(X_fold_train, Y_fold_train, P, C);
            
            % Validate - using classification error as objective per assignment
            predictions = predictCustomMLP(Mdl, X_fold_val);
            fold_errors(k) = sum(predictions ~= Y_fold_val) / length(Y_fold_val);
        end
        
        % Average validation error across folds - Pattern from Professor's PolynomialFitCrossValidation.m
        cv_errors(i, j) = mean(fold_errors);
    end
    fprintf('\n');
    
    % Select optimal P - Pattern from Professor's PolynomialFitCrossValidation.m
    [min_error, best_idx] = min(cv_errors(i, :));
    P_optimal(i) = P_values(best_idx);
    
    fprintf('  Cross-validation errors: ');
    fprintf('%.4f ', cv_errors(i, :));
    fprintf('\n');
    fprintf('  Optimal P* = %d (CV error = %.4f)\n', P_optimal(i), min_error);
    
    % Train final model with optimal P on full training set
    % Multiple random initializations to avoid local optima per assignment requirement
    fprintf('  Final model with P* = %d\n', P_optimal(i));
    
    best_log_likelihood = -inf;
    best_model = [];
    n_initializations = 3;
    
    for init = 1:n_initializations
        % Set seed for reproducible but distinct initializations - Recommended by ChatGPT
        rng(5644 + init, 'twister');
        
        % Train with random initialization
        Mdl_final = trainCustomMLP(X_current, Y_current, P_optimal(i), C);
        
        % Calculate training log-likelihood to select best initialization
        posterior_probs = mlpModelClassification(X_current, Mdl_final.params);
        
        log_likelihood = 0;
        for n_sample = 1:length(Y_current)
            true_class = Y_current(n_sample);
            % Add small epsilon to avoid log(0)
            log_likelihood = log_likelihood + log(posterior_probs(true_class, n_sample) + 1e-10);
        end
        
        % Keep model with highest log-likelihood
        if log_likelihood > best_log_likelihood
            best_log_likelihood = log_likelihood;
            best_model = Mdl_final;
        end
    end
    
    trained_models{i} = best_model;
    fprintf('  Best model selected (log-likelihood = %.4f)\n', best_log_likelihood);
    
    % Report training error for the best model
    train_predictions = predictCustomMLP(best_model, X_current);
    train_error = sum(train_predictions ~= Y_current) / length(Y_current);
    fprintf('  Training P(error): %.4f\n', train_error);
end

% Report Structure Adapted from discussions with Claude
fprintf('\n');
fprintf('MODEL ORDER SELECTION SUMMARY\n');
fprintf('  Training Size  |  Optimal P*  |  CV Error\n');
fprintf('--------------------------------------------------------------------\n');
for i = 1:length(N_train)
    fprintf('     %5d       |      %2d      |   %.4f\n', ...
            N_train(i), P_optimal(i), cv_errors(i, find(P_values == P_optimal(i))));
end

% EVALUATE MLPS ON TEST SET
% Report Structure Adapted from discussions with Claude
fprintf('\n EVALUATING TRAINED MLPS ON TEST SET\n');
% Storage for results
P_error_MLP = zeros(length(N_train), 1);

% Evaluate each trained model on test set
for i = 1:length(N_train)
    fprintf('N = %5d (P* = %2d): ', N_train(i), P_optimal(i));
    
    % Get posterior probabilities and predictions from MLP
    posterior_probs = mlpModelClassification(x_test, trained_models{i}.params);
    
    % Apply MAP decision rule (argmax of posteriors)
    [~, map_decisions] = max(posterior_probs, [], 1);
    
    % Calculate P(error)
    P_error_MLP(i) = sum(map_decisions ~= labels_test) / N_test;
    
    fprintf('P(error) = %.4f (%.2f%%)\n', P_error_MLP(i), 100*P_error_MLP(i));
end

fprintf('\n');
fprintf('PERFORMANCE COMPARISON: MLP vs OPTIMAL\n');
fprintf('  Training Size  |  P*  |  MLP P(error)  |  Optimal P(error)  |  Gap\n');
fprintf('--------------------------------------------------------------------\n');
for i = 1:length(N_train)
    gap = P_error_MLP(i) - P_error_optimal;
    fprintf('     %5d       |  %2d  |     %.4f      |       %.4f        | %+.4f\n', ...
            N_train(i), P_optimal(i), P_error_MLP(i), P_error_optimal, gap);
end

% P(error) vs Training Set Size
figure(3), clf,
semilogx(N_train, P_error_MLP, 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on,
semilogx(N_train, P_error_optimal * ones(size(N_train)), 'r--', 'LineWidth', 2);
xlabel('Number of Training Samples');
ylabel('Probability of Error');
title('MLP Classifier Performance vs Training Set Size');
legend('Trained MLP', sprintf('Theoretical Optimal (%.4f)', P_error_optimal), ...
       'Location', 'northeast');
grid on;

% Add annotations for P* values
for i = 1:length(N_train)
    text(N_train(i), P_error_MLP(i) + 0.01, sprintf('P*=%d', P_optimal(i)), ...
         'FontSize', 8, 'HorizontalAlignment', 'center');
end

% Figure 4: Cross-validation error vs P for each training size
figure(4), clf,
for i = 1:length(N_train)
    subplot(2, 3, i);
    plot(P_values, cv_errors(i, :), 'b-o', 'LineWidth', 2, 'MarkerSize', 6); hold on,
    
    % Mark optimal P
    best_idx = find(P_values == P_optimal(i));
    plot(P_optimal(i), cv_errors(i, best_idx), ...
         'ro', 'MarkerSize', 10, 'LineWidth', 2);
    
    xlabel('Number of Perceptrons (P)');
    ylabel('10-Fold CV Error');
    title(sprintf('N = %d', N_train(i)));
    grid on;
    legend('CV Error', sprintf('P* = %d', P_optimal(i)), 'Location', 'best');
end
sgtitle('Model Order Selection via Cross-Validation');

fprintf('FINAL SUMMARY:\n');
fprintf('  Theoretical optimal P(error): %.4f (%.2f%%)\n', P_error_optimal, 100*P_error_optimal);
fprintf('\n  MLP Results:\n');
for i = 1:length(N_train)
    fprintf('    N=%5d: P*=%2d, P(error)=%.4f (gap: %+.4f)\n', ...
            N_train(i), P_optimal(i), P_error_MLP(i), P_error_MLP(i) - P_error_optimal);
end
fprintf('\n  Best MLP: N=%d with P*=%d achieves P(error)=%.4f\n', ...
        N_train(find(P_error_MLP == min(P_error_MLP))), ...
        P_optimal(find(P_error_MLP == min(P_error_MLP))), ...
        min(P_error_MLP));
fprintf('\n  Observation: MLP performance approaches optimal as N increases\n\n');


% HELPER FUNCTIONS
% Custom MLP training function
% Developed with Claude AI, integrating Professor's MLP patterns from mleMLPwAWGN.m and kfoldMLP.m
function model = trainCustomMLP(X, Y, nPerceptrons, nClasses)
    % X: n x N (features x samples)
    % Y: 1 x N (class labels 1 to nClasses)
    % nPerceptrons: number of hidden layer neurons
    % nClasses: number of output classes
    
    [nX, N] = size(X);
    nY = nClasses;
    
    % Xavier/Glorot-style initialization for better starting point
    % Recommended by ChatGPT for numerical stability
    params.A = randn(nPerceptrons, nX) * sqrt(2/nX);
    params.b = zeros(nPerceptrons, 1);  % Start biases at zero
    params.C = randn(nY, nPerceptrons) * sqrt(2/nPerceptrons);
    params.d = zeros(nY, 1);
    
    % Convert to vector for fminsearch - Pattern from Professor's mleMLPwAWGN.m
    vecParamsInit = [params.A(:); params.b; params.C(:); params.d];
    
    % Define sizeParams BEFORE using it in fminsearch
    sizeParams = [nX; nPerceptrons; nY];
    
    % Scale optimization effort with model complexity
    nParams = length(vecParamsInit);
    maxEvals = max(5000, 50 * nParams);  % At least 50 evaluations per parameter
    maxIter = max(2000, 20 * nParams);   % At least 20 iterations per parameter
    
    % Optimize using fminsearch - Pattern from Professor's mleMLPwAWGN.m
    % fminsearch minimizes cross-entropy loss (ML estimation per assignment)
    options = optimset('MaxFunEvals', maxEvals, ...
                       'MaxIter', maxIter, ...
                       'TolFun', 1e-6, ...  % Tighter convergence tolerance
                       'TolX', 1e-6, ...
                       'Display', 'off');
    
    vecParams = fminsearch(@(vecParams)(objectiveFunctionClassification(X, Y, sizeParams, vecParams)), ...
                           vecParamsInit, options);
    
    % Convert back to structure - Pattern from Professor's mleMLPwAWGN.m 
    params.A = reshape(vecParams(1:nX*nPerceptrons), nPerceptrons, nX);
    params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
    params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons), nY, nPerceptrons);
    params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
    
    model.params = params;
    model.nPerceptrons = nPerceptrons;
    model.nClasses = nClasses;
end


% MLP forward pass with numerically stable softmax output
% Adapted from Professor's mleMLPwAWGN.m mlpModel function with softmax developed with Claude AI
% Numerical stability improvement recommended by ChatGPT
function H = mlpModelClassification(X, params)
    % X: n x N (features x samples)
    % H: nY x N (class posterior probabilities)
    
    N = size(X, 2);
    nY = length(params.d);
    
    % First layer: linear transformation - From Professor's mleMLPwAWGN.m mlpModel
    U = params.A * X + repmat(params.b, 1, N);  % nPerceptrons x N
    
    % Hidden layer: ISRU activation - From Professor's kfoldMLP.m and mleMLPwAWGN.m
    Z = activationFunction(U);  % nPerceptrons x N
    
    % Second layer: linear transformation - From Professor's mleMLPwAWGN.m mlpModel
    V = params.C * Z + repmat(params.d, 1, N);  % nY x N
    
    % Output layer: numerically stable softmax - Recommended by ChatGPT
    % Max-subtraction prevents overflow for large V values
    Vmax = max(V, [], 1);  % 1 x N
    E = exp(V - repmat(Vmax, nY, 1));  % nY x N
    H = E ./ repmat(sum(E, 1), nY, 1);  % nY x N (class posteriors sum to 1)
end

% ISRU activation function - from Professor's kfoldMLP.m and mleMLPwAWGN.m
function out = activationFunction(in)
    out = in ./ sqrt(1 + in.^2);  % ISRU - smooth ramp-style nonlinearity
end

% Objective function: cross-entropy loss
% Adapted from Professor's mleMLPwAWGN.m objectiveFunction with cross-entropy developed with Claude AI
function objFncValue = objectiveFunctionClassification(X, Y, sizeParams, vecParams)
    % X: n x N
    % Y: 1 x N (class labels 1 to nClasses)
    % Returns: average cross-entropy loss (ML estimation per assignment)
    
    N = size(X, 2);
    nX = sizeParams(1);
    nPerceptrons = sizeParams(2);
    nY = sizeParams(3);
    
    % Convert vector to structure - Pattern from Professor's mleMLPwAWGN.m objectiveFunction
    params.A = reshape(vecParams(1:nX*nPerceptrons), nPerceptrons, nX);
    params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
    params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons), nY, nPerceptrons);
    params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
    
    % Forward pass
    H = mlpModelClassification(X, params);  % nY x N (class posteriors)
    
    % Cross-entropy loss 
    % ML estimation minimizes average cross-entropy loss
    crossEntropy = 0;
    for i = 1:N
        trueClass = Y(i);
        % Add epsilon to avoid log(0)
        crossEntropy = crossEntropy - log(H(trueClass, i) + 1e-10);
    end
    objFncValue = crossEntropy / N;
end

% Prediction function
function predictions = predictCustomMLP(model, X)
    % X: n x N
    % predictions: 1 x N (class labels)
    
    % Get posterior probabilities
    H = mlpModelClassification(X, model.params);
    
    % MAP decision rule
    [~, predictions] = max(H, [], 1);
end


% Gaussian evaluation with Cholesky factorization for numerical stability
% Adapted from Professor's evalGaussian.m with Cholesky improvement recommended by ChatGPT
function g = evalGaussian(x, mu, Sigma)
    [n, N] = size(x);
    R = chol(Sigma, 'lower');  % Cholesky factorization: Sigma = R*R'
    Xc = x - repmat(mu, 1, N);  % Center the data
    Y = R \ Xc;  % Solve R*Y = Xc via forward substitution
    E = -0.5 * sum((R' \ Y).^2, 1);  % Mahalanobis distance via back substitution
    C = (2*pi)^(-n/2) / prod(diag(R));  % Normalization constant
    g = C * exp(E);
end

% Code for generateDataFromGMM function taken from Professor's generateDataFromGMM.m
function [x,labels] = generateDataFromGMM(N,gmmParameters)
priors = gmmParameters.priors;
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1);
C = length(priors);
x = zeros(n,N); labels = zeros(1,N); 
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
    indl = find(u <= thresholds(l)); Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl);
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end
end