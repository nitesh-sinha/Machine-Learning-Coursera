function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m,1) X];

for i = 1:m
  % Hidden layer activations
  a1 = X(i,:);
  a1 = a1';
  z2 = Theta1 * a1; % 25 X 1 vector
  a2 = sigmoid(z2); % 25 X 1 vector
  a2 = [1; a2]; % Add a bias unit so now size of a2 is 26 X 1

  % Output layer activations
  z3 = Theta2 * a2; % 10 X 1 vector
  a3 = sigmoid(z3);
  % Compute max along row(since we need the output to be 1 at the 
  % index which denotes the digit number and 0's at all other indices)
  [maximum, index] = max(a3); 
  p(i) = index;
endfor

% fprintf("size of output vector a3: %f %f \n", size(a3));


% =========================================================================


end
