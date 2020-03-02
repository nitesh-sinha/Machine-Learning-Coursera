function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

[nonRegCost, nonRegGradient] = costFunction(theta, X, y);


%fprintf('Non-reg Cost at initial theta (zeros): %f\n', nonRegCost);
%fprintf('Theta values: %f\n', theta);
%pause;

% Lets regularize the cost and gradient now
n = size(X,2);
SumThetaSquares = 0;
for j = 2:n % skip first value in theta which is theta(1)
  SumThetaSquares =  SumThetaSquares + theta(j) ^ 2;
endfor

%fprintf('SumThetaSquares value: %f\n', SumThetaSquares);


% Add the regularization term to the non regularized cost & gradient
regCost = nonRegCost + lambda/(2*m) * SumThetaSquares;
% Don't regularize first gradient value
regGradient(1) = nonRegGradient(1);
for j = 2:n
  regGradient(j) = nonRegGradient(j) + (lambda/m) * theta(j);
endfor

J = regCost;
grad = regGradient; 

% =============================================================

end
