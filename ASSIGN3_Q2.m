%{
                    EECE5644 FALL 2025 - ASSIGNMENT 3
                                QUESTION 2 
        GMM Model Order Selection using 10-Fold Cross-Validation
%}

clear all, close all,

fprintf('   ASSIGNMENT 3 - QUESTION 2\n');
% Set seed for reproducibility
rng(5644);

% DESIGN TRUE GMM (M=4 components in 2D)
fprintf('TRUE GMM DESIGN\n');

% True model has M=4 Gaussian components
M_true = 4;
n = 2;  % 2-dimensional data

% Mixing probabilities 
alpha_true = [0.40, 0.30, 0.20, 0.10];

% Mean vectors 
mu_true = zeros(n, M_true);
mu_true(:,1) = [0; 0];      % Component 1: Origin (overlap with 2)
mu_true(:,2) = [1.5; 1.5];  
mu_true(:,3) = [-4; -4];    
mu_true(:,4) = [4; -4];     

% Covariance matrices 
Sigma_true = zeros(n, n, M_true);
Sigma_true(:,:,1) = [1.6  0.4;   0.4  1.2];  
Sigma_true(:,:,2) = [1.1 -0.2;  -0.2  1.8];  
Sigma_true(:,:,3) = [0.7  0.1;   0.1  0.5];  
Sigma_true(:,:,4) = [1.3  0.0;   0.0  0.6];  

% Verify overlap criterion for components 1 and 2 - Added based on feedback
% from Chatgpt
dist_12 = norm(mu_true(:,1) - mu_true(:,2));
eig_1 = eig(Sigma_true(:,:,1));
eig_2 = eig(Sigma_true(:,:,2));
avg_eig_sum = mean(eig_1) + mean(eig_2);

fprintf('True GMM Configuration:\n');
fprintf('  Number of components: M = %d\n', M_true);
fprintf('  Dimension: n = %d\n', n);
fprintf('  Mixing probabilities: [%.2f, %.2f, %.2f, %.2f] \n', alpha_true);
fprintf('\n  Mean Vectors:\n');
for m = 1:M_true
    fprintf('    Component %d: [%.1f, %.1f]\n', m, mu_true(1,m), mu_true(2,m));
end
fprintf('\n  Covariance Matrices:\n');
for m = 1:M_true
    fprintf('    Component %d: [%.1f %.1f; %.1f %.1f]\n', m, ...
        Sigma_true(1,1,m), Sigma_true(1,2,m), ...
        Sigma_true(2,1,m), Sigma_true(2,2,m));
end
fprintf('\nOverlap Verification (Components 1 & 2):\n');
fprintf('  Distance between means: %.3f\n', dist_12);
fprintf('  Sum of avg eigenvalues: %.3f\n', avg_eig_sum);
fprintf('  Ratio (dist/eigenvalue_sum): %.3f\n', dist_12/avg_eig_sum);
if dist_12 <= avg_eig_sum
    fprintf('There is significant overlap between components 1 & 2 as required\n');
else
    fprintf(' Not overlapping\n');
end

% Package parameters for generateDataFromGMM
gmm_true.priors = alpha_true;
gmm_true.meanVectors = mu_true;
gmm_true.covMatrices = Sigma_true;

% Visualize true GMM
N_vis = 1000;
[x_vis, labels_vis] = generateDataFromGMM(N_vis, gmm_true);

figure(1), clf;
colors = ['r', 'b', 'g', 'm'];
hold on;
for m = 1:M_true
    idx = (labels_vis == m);
    plot(x_vis(1,idx), x_vis(2,idx), [colors(m) '.'], 'MarkerSize', 8);
end
axis equal, grid on;
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
title('True GMM Distribution (M=4, with Components 1 & 2 Overlapping)');
legend('Component 1', 'Component 2', 'Component 3', 'Component 4', ...
    'Location', 'best');

% MODEL ORDER SELECTION
fprintf('\nMODEL ORDER SELECTION\n');

% Experimental parameters
N_sizes = [10, 100, 1000];       % Dataset sizes
n_sizes = length(N_sizes);
M_candidates = 1:10;             % Candidate model orders
n_trials = 100;                  % Number of experimental trials
K_folds = 10;                    % K-fold cross-validation

fprintf('Experiment Configuration:\n');
fprintf('  Dataset sizes: N = [%s]\n', num2str(N_sizes));
fprintf('  Candidate orders: M = 1 to %d\n', max(M_candidates));
fprintf('  Number of trials: %d\n', n_trials);
fprintf('  Cross-validation folds: K = %d\n\n', K_folds);

% Storage for results
M_selected = zeros(n_trials, n_sizes);  % Selected model order for each trial

% Suppress expected convergence warnings for high-order GMMs with small N
% Added based on suggestion from Claude
warning('off', 'stats:gmdistribution:FailedToConverge');


% Progress indicator - Added based on suggestion from Claude
fprintf('Progress indicator\n');
for trial = 1:n_trials
    if mod(trial, 10) == 0
        fprintf('  Completed %d/%d trials\n', trial, n_trials);
    end
    
    % For each dataset size
    for n_idx = 1:n_sizes
        N_current = N_sizes(n_idx);
        
        % Generate dataset - Based on generateDataFromGMM.m
        [x_data, ~] = generateDataFromGMM(N_current, gmm_true);
        
        % Randomize data order before CV partitioning - Based on suggestion
        % from Chatgpt
        rand_idx = randperm(N_current);
        x_data = x_data(:, rand_idx);


        % Perform 10-fold cross-validation for each candidate M
        % Structure adapted from PolynomialFitCrossValidation.m
        avg_logL_val = zeros(1, length(M_candidates));
        
        for m_idx = 1:length(M_candidates)
            M_candidate = M_candidates(m_idx);
            
            % K-fold cross-validation
            logL_val_folds = zeros(1, K_folds);
            
            % Create fold indices
            if N_current >= K_folds
                fold_limits = round(linspace(0, N_current, K_folds + 1));
            else
                % For N < K, use leave-one-out
                fold_limits = 0:N_current;
            end
            
            for k = 1:(length(fold_limits) - 1)
                % Partition data into train and validation sets
                idx_val = (fold_limits(k)+1):fold_limits(k+1);
                idx_train = setdiff(1:N_current, idx_val);
                
                x_train = x_data(:, idx_train);
                x_val = x_data(:, idx_val);
                
                % Train GMM using EM algorithm (fitgmdist) - Modified based
                % on feedback from Claude for faster convergence                
                % Regularization prevents singular covariances
                try
                    opts = statset('MaxIter', 300, 'Display', 'off');  % Reduce iterations
                    gmm_model = fitgmdist(x_train', M_candidate, 'Start', 'plus',...              
                    'RegularizationValue', 1e-3, 'Options', opts);
                    
                    % Compute validation log-likelihood
                    logL_val_folds(k) = sum(log(pdf(gmm_model, x_val') + 1e-10));
                catch
                    % If fitting fails, assign very poor log-likelihood
                    logL_val_folds(k) = -1e10;
                end
            end
            
            % Average validation log-likelihood across folds
            avg_logL_val(m_idx) = mean(logL_val_folds);
        end
        
        % Select model order with highest average validation log-likelihood
        [~, best_m_idx] = max(avg_logL_val);
        M_selected(trial, n_idx) = M_candidates(best_m_idx);

        % Save CV curve from first trial at N=1000 for visualization
        if trial == 1 && n_idx == 3
            cv_curves_N1000 = avg_logL_val;
        end
    end
end

% Re-enable convergence warnings
warning('on', 'stats:gmdistribution:FailedToConverge');

% Results Analysis - Adapted from discussion with Chatgpt
fprintf('\nRESULTS ANALYSIS\n');

% Calculate selection rates for each (M, N) combination
selection_rates = zeros(length(M_candidates), n_sizes);

for n_idx = 1:n_sizes
    for m_idx = 1:length(M_candidates)
        M_current = M_candidates(m_idx);
        selection_rates(m_idx, n_idx) = sum(M_selected(:, n_idx) == M_current) / n_trials;
    end
end

% Display selection rate table
fprintf('Selection Rate Table (Percentage of %d trials):\n\n', n_trials);
fprintf('    M  |  N=10   | N=100  | N=1000 |\n');
fprintf('-------|---------|--------|--------|\n');
for m_idx = 1:length(M_candidates)
    fprintf('   %2d  |  %5.1f%% | %5.1f%% | %5.1f%% |\n', ...
        M_candidates(m_idx), ...
        selection_rates(m_idx, 1)*100, ...
        selection_rates(m_idx, 2)*100, ...
        selection_rates(m_idx, 3)*100);
end
fprintf('\n');

% True model order
fprintf('True Model Order: M = %d\n', M_true);
fprintf('Selection rate for M=%d:\n', M_true);
for n_idx = 1:n_sizes
    fprintf('  N=%4d: %.1f%%\n', N_sizes(n_idx), ...
        selection_rates(M_true, n_idx)*100);
end
fprintf('\n');

% Figure 2: Selection rate - Adapted from discussion with Claude
figure(2), clf;
M_range_display = 1:10;
rate_subset = selection_rates(M_range_display, :);
x_positions = M_range_display;

% Create grouped bar chart
b = bar(x_positions, rate_subset', 'grouped');
colors_N = [0.8 0.2 0.2; 0.2 0.6 0.8; 0.2 0.8 0.2];  % Red, Blue, Green
for i = 1:n_sizes
    b(i).FaceColor = colors_N(i, :);
end

hold on;
xline(M_true, 'k--', 'LineWidth', 2.5);
text(M_true, 0.95, sprintf('True M=%d', M_true), ...
    'FontSize', 11, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');

ylabel('Selection Rate', 'FontSize', 12);
xlabel('Model Order (M)', 'FontSize', 12);
title('GMM Model Order Selection Rates Across Dataset Sizes');
legend({'N=10', 'N=100', 'N=1000'}, 'Location', 'northeast', 'FontSize', 11);
ylim([0 1]);
xlim([0.5, 10.5]);
xticks(1:10);  % Force x-axis to show 1,2,3,...,10
grid on;

% Figure 3: Evolution of P(M*=4) with N
figure(3), clf;
plot(N_sizes, selection_rates(M_true, :), 'bo-', 'LineWidth', 2, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'b');
xlabel('Number of Samples (N)', 'FontSize', 12);
ylabel(sprintf('P(M^* = %d | Data)', M_true), 'FontSize', 12);
title(sprintf('Probability of Selecting True Model Order (M=%d)', M_true));
grid on;
set(gca, 'XScale', 'log');
ylim([0 1]);

% Add percentage labels
for n_idx = 1:n_sizes
    text(N_sizes(n_idx), selection_rates(M_true, n_idx) + 0.05, ...
        sprintf('%.1f%%', selection_rates(M_true, n_idx)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

% Figure 4: Cross-Validation curve - Added based on suggestion from Chatgpt
figure(4), clf;
% Plot validation log-likelihood vs M 
plot(M_candidates, cv_curves_N1000, 'b-o', 'LineWidth', 2.5, ...
     'MarkerSize', 9, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
hold on;

% Mark the selected M 
[~, selected_idx] = max(cv_curves_N1000);
plot(M_candidates(selected_idx), cv_curves_N1000(selected_idx), ...
     'rs', 'MarkerSize', 15, 'LineWidth', 3, 'MarkerFaceColor', 'r');

% Mark true M 
plot(M_true, cv_curves_N1000(M_true), 'g^', ...
     'MarkerSize', 15, 'LineWidth', 2.5, ...
      'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g');

xlabel('Model Order (M)', 'FontSize', 12);
ylabel('Average Validation Log-Likelihood', 'FontSize', 12);
title('10-Fold Cross-Validation Curve (N=1000)');
legend('CV Log-Likelihood', ...
       sprintf('Selected M*=%d', M_candidates(selected_idx)), ...
       sprintf('True M=%d', M_true), ...
       'Location', 'southeast', 'FontSize', 11);
grid on;
xlim([0.5, 10.5]);

% SUMMARY
fprintf('\nDATA SUMMARY\n');

% Mean and mode of selected orders
fprintf('Selected Model Order Statistics:\n');
fprintf('Dataset Size | Mean M* | Mode M* | Std Dev |\n');
fprintf('-------------|---------|---------|---------|\n');
for n_idx = 1:n_sizes
    mean_M = mean(M_selected(:, n_idx));
    mode_M = mode(M_selected(:, n_idx));
    std_M = std(M_selected(:, n_idx));
    fprintf('   N=%4d    |  %.2f   |   %2d    |  %.2f   |\n', ...
        N_sizes(n_idx), mean_M, mode_M, std_M);
end
fprintf('\n');

% Probability of selecting M <= M_true
fprintf('Cumulative Selection Probabilities:\n');
fprintf('  P(M* <= %d):\n', M_true);
for n_idx = 1:n_sizes
    prob_underfit = sum(selection_rates(1:M_true, n_idx));
    fprintf('    N=%4d: %.3f\n', N_sizes(n_idx), prob_underfit);
end
fprintf('\n');

% Probability of selecting M > M_true
fprintf('  P(M* > %d):\n', M_true);
for n_idx = 1:n_sizes
    prob_overfit = sum(selection_rates((M_true+1):end, n_idx));
    fprintf('    N=%4d: %.3f\n', N_sizes(n_idx), prob_overfit);
end
fprintf('\n');

% FINAL SUMMARY
fprintf('KEY FINDINGS:\n');
fprintf('1. True model order: M = %d\n', M_true);
fprintf('2. Selection accuracy improves with N:\n');
for n_idx = 1:n_sizes
    fprintf('   N=%4d: %.1f%% correct\n', N_sizes(n_idx), ...
        selection_rates(M_true, n_idx)*100);
end
fprintf('\n3. Overlap between components 1 & 2 makes\n');
fprintf('   model selection challenging with small N\n');
fprintf('\n4. Cross-validation successfully recovers\n');
fprintf('   true model order as sample size increases\n');

% HELPER FUNCTIONS
% Code for generateDataFromGMM function taken from Professor's generateDataFromGMM.m 
function [x, labels] = generateDataFromGMM(N, gmmParameters)
    priors = gmmParameters.priors;
    meanVectors = gmmParameters.meanVectors;
    covMatrices = gmmParameters.covMatrices;
    n = size(gmmParameters.meanVectors, 1);
    C = length(priors);
    x = zeros(n, N);
    labels = zeros(1, N);
    u = rand(1, N);
    thresholds = [cumsum(priors), 1];
    for l = 1:C
        indl = find(u <= thresholds(l));
        Nl = length(indl);
        labels(1, indl) = l * ones(1, Nl);
        u(1, indl) = 1.1 * ones(1, Nl);
        x(:, indl) = mvnrnd(meanVectors(:, l), covMatrices(:, :, l), Nl)';
    end
end