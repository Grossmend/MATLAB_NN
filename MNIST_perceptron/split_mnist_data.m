
function [trainData, trainTarget, testData, testTarget] = split_mnist_data(data, trainPercent)

% function split data into trainData, trainTarget, testData and testTarget
% Grossmend, 2018

    if trainPercent <= 0 || trainPercent >= 1
        error('trainPercent must be between 0 and 1')
    end
    
    N = round(trainPercent * size(data,1));
    r = randperm(size(data,1));

    trainData = data(r(1:N),2:end);
    trainTarget = data(r(1:N),1);
    
    
    testData = data(r(N+1:end),2:end);
    testTarget = data(r(N+1:end),1);

    if isempty(trainData) || ...
            isempty(trainTarget) || ...
            isempty(testData) ||...
            isempty(testTarget)
        error('Data for neural train, test or targets is empty')
    end
    
end
