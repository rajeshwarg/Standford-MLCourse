function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

Clist = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmalist = [0.01 0.03 0.1 0.3 1 3 10 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
besterror = length(X) + length(Xval) + 100;

for i = 1:length(Clist)
    C_val = Clist(i);
    for j = 1:length(sigmalist)
        sigma_val = sigmalist(j);
        model = svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val)); 
        predictions = svmPredict(model, Xval);
        error_val =  mean(double(predictions ~= yval));
        if (error_val < besterror)
            besterror = error_val;
            C = C_val;
            sigma = sigma_val;
        endif
    end
end

% =========================================================================

end
