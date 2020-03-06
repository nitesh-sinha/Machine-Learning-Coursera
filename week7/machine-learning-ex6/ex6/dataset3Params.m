function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set.
%

C_vals = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vals = [0.01 0.03 0.1 0.3 1 3 10 30];
% Store model params(C, sigma) and prediction error
%model_params_and_error = zeros((size(C_vals,2) * size(sigma_vals,2)), 3); 
model_params_and_error = [];

for i = 1:length(C_vals)
  for j = 1:length(sigma_vals)
    model = svmTrain(X, y, C_vals(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vals(j)));
    predictions = svmPredict(model, Xval);
    % Calculate prediction error(i.e. fractions of dataset classified incorrectly) on CV set
    pred_error = mean(double(predictions ~= yval));
    %fprintf("Using param C = %f and sigma + %f to train SVM. Error is: %f \n", C_vals(i), sigma_vals(j), pred_error);
    model_params_and_error = [model_params_and_error; [C_vals(i) sigma_vals(j) pred_error]];
  endfor
endfor

% Calculate lowest prediction error and corresponding model params
[min_error indx] = min(model_params_and_error(:,3));
%fprintf("Min prediction error on CV set is: %f \n. Corresponding model params are: C= %f and sigma = %f", 
%        min_error, model_params_and_error(indx, 1), model_params_and_error(indx, 2));
        
C = model_params_and_error(indx, 1);
sigma = model_params_and_error(indx, 2);

% =========================================================================

end
