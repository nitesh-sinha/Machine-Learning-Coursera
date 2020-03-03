function [error_train, error_val] = ...    randomLearningCurve(X, y, Xval, yval, lambda)%RANDOMLEARNINGCURVE Generates the train and cross validation set errors needed %to plot a learning curve generated from randomly selected input examples.%   [error_train, error_val] = ...%       RANDOMLEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and%       cross validation set errors for a learning curve generated from randomly%       selected input examples.. In particular, %       it returns two vectors of the same length - error_train and %       error_val. Then, error_train(i) contains the training error for%       i examples (and similarly for error_val(i)).%%   In this function, you will compute the train and test errors for%   dataset sizes from 1 up to m. %   In practice, when working with larger datasets, you might want to do this %   in larger intervals of number of input examples.%   For each dataset of size "s", you should randomly%   select the input examples upto a certain number of times(say 50) and then%   averaged error should be used to store in error_train and error_val.%   %% Number of training examplesm = size(X, 1);% Number of validation set examplesv = size(Xval, 1);% Number of random selectionsnumRandIter = 50;% You need to return these values correctlyerror_train = zeros(m, 1);error_val   = zeros(m, 1);% ====================== YOUR CODE HERE ======================% Instructions: Fill in this function to return training errors in %               error_train and the cross validation errors in error_val. %               i.e., error_train(i) and %               error_val(i) should give you the average errors%               obtained after training on randomly selected i examples.%% Note: You should evaluate the training error on the first i randonly%       selected training examples (i.e., X(1:i, :) and y(1:i)).%%       For the cross-validation error, we will similarly evaluate on%       the first i CV set examples.%% ---------------------- Sample Solution ----------------------for i = 1:m  fprintf("For iteration %f \n", i);  % For every i, iterate numRandIter times and then calculate average errors  error_train_random = zeros(numRandIter, 1);  error_val_random   = zeros(numRandIter, 1);  for j = 1:numRandIter    randNumsTrain = randperm(m);    randNumsVal = randperm(v);    % Grab the first i elements from random generated numbers    randRowNumsTrain = randNumsTrain(1:i);    randRowNumsVal = randNumsVal(1:i);    % Now grab those rows from X and y whose row numbers correspond to    % the elements in randRowNums        subset_x = X(randRowNumsTrain, :);    subset_y = y(randRowNumsTrain);    subset_xval = Xval(randRowNumsVal, :);    subset_yval = yval(randRowNumsVal);    computed_theta = trainLinearReg(subset_x, subset_y, lambda);      % Now calculate errors    % Train error on first i training examples    % Cross validation error on subset of i examples of CV set    error_train_random(j) = linearRegCostFunction(subset_x, subset_y, computed_theta, 0); % Set lambda = 0(no regularizarion)    error_val_random(j) = linearRegCostFunction(subset_xval, subset_yval, computed_theta, 0);  endfor      % Now calculate the average errors and store them    error_train(i) = mean(error_train_random);    error_val(i) = mean(error_val_random);   endfor% -------------------------------------------------------------% =========================================================================end