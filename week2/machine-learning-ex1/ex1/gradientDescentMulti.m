function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    predictedData = X*theta; % mX1 vector
    predictionErrors = predictedData - y; % mX1 vector
    
    temp = predictionErrors' * X ;% 1X(n+1) matrix where n= no. of features in input
    theta = theta - (alpha/m)*temp';
    
    % compute the cost using obatined theta params
    cost = computeCostMulti(X, y, theta);
    %fprintf('Iteration = %f  ,   current cost = %f \n', iter, cost);
    if iter >= 2,
      prevCost = J_history(iter-1);
      if cost > prevCost, % linear regression is not converging anymore. So stop!
        break;
      endif
    endif

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
