function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%



initial_theta = zeros(n+1,1);
temp_theta = [];
options = optimset('GradObj', 'on', 'MaxIter', 100);
for i = 1:num_labels
  % Calculate separate optimum theta params for all labels
  % Note y==i returns a vector of size same as "y" but with 1's at indices 
  % where y is equal to i; 0's elsewhere
  [optimumTheta] = fmincg(@(t)(lrCostFunction(t, X, (y==i), lambda)), initial_theta, options);
  % fprintf('size of optimum theta: %f %f \n', size(optimumTheta)); % size = [n+1 1]
  temp_theta = [temp_theta optimumTheta]; 
endfor

% Since temp_theta is of size "(n+1) X num_labels", we transpose it to obtain all_theta
all_theta = temp_theta';


% Essentially, in one-vs-all classification problem(assuming K output classes), we do the following:
%   1. Choose initial theta and train classifier for 1st label being true and rest being false.
%   2. Choose initial theta and train classifier for 2nd label being true and rest being false.
%   3. Similarly choose initial thetas and train classifiers for remaining labels being true and rest being false(one at a time).
%   4. Finally we have K classifier ouputs(optimal theta values), one for each label.
%   5. Choose 1st classifier output(theta values), use hypothesis equation(maybe g(X*theta)) and calculate output mX1 size hypothesis(say H1).
%   6. Choose 2nd classifier output(theta values), use hypothesis equation(maybe g(X*theta)) and calculate output mX1 size hypothesis(say H2).
%   7. Similarly choose all remaining (K-2) classifier outputs, and use hypothesis equation(maybe g(X*theta)) and 
%           calculate output hypotheses(say H3, H4,....Hk), each of size mX1.
%   8. Now out of K output hypotheses we have obtained for the first sample, choose the one which is the largest(i.e. probability being highest).
%   9. Similarly repeat step 8 for all other (m-1) samples.
% =========================================================================


end
