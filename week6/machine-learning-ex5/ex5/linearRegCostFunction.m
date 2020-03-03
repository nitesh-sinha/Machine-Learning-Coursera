function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Let's calculate the cost
hypothesis = X * theta;
errors = hypothesis - y;
% sumSqErrors = sum(errors .^2);
sumSqErrors = errors' * errors;
unregCost = 1/(2*m) * sumSqErrors;
% fprintf("Unreg cost: %f \n", unregCost);
regularizationCost = lambda/(2*m) * sum(theta(2:end).^2);
regCost = unregCost + regularizationCost;

% Let's calculate gradient
unregGrad = 1/m * errors' * X;
unregGrad = unregGrad(:);
% Skip regularizing first term
temp = unregGrad(1);
regGrad = unregGrad + (lambda/m)*theta;
regGrad(1) = temp;


J = regCost;
grad = regGrad;

% =========================================================================

grad = grad(:);

end
