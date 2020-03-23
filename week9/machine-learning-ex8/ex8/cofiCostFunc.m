function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the X and Theta matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


predicted = X * Theta';
% Nullify the entries which were not actually rated by users
predictedToConsider = predicted .* R;
predErrors = predictedToConsider - Y;
sqPredErrors = predErrors .^ 2;
unregCost = 1/2 * sum(sum(sqPredErrors));

% Lets calculate gradients
unreg_X_grad = predErrors * Theta;
unreg_Theta_grad = predErrors' * X;

% Adding regularization to cost function
SqTheta = Theta .^ 2;
ThetaReg = sum(sum(SqTheta));
SqFeature = X .^ 2;
FeatureReg = sum(sum(SqFeature));
regCost = lambda/2 * (ThetaReg + FeatureReg);

% Adding regularization to gradient terms
regterm_X_grad = lambda * X;
regterm_Theta_grad = lambda * Theta;



J = unregCost + regCost;
X_grad = unreg_X_grad;
Theta_grad = unreg_Theta_grad + regterm_Theta_grad;
X_grad = unreg_X_grad + regterm_X_grad;
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
