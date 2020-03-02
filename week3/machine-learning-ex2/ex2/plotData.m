function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

onesPos = find(y==1); % Returns index nos. in a vector where value equals 1
zerosPos = find(y==0);

plot(X(onesPos,1), X(onesPos,2), 'k+', "linewidth", 2, "markersize", 10);
%scatter(X(onesPos,1), X(onesPos,2), "k", "marker", '+');
hold on;
plot(X(zerosPos,1), X(zerosPos,2), 'ko', "markerfacecolor", "y", "markersize", 10);
%scatter(X(zerosPos,1), X(zerosPos,2), "y", "marker", 'o', "filled");





% =========================================================================



hold off;

end
