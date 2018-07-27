
% ��������� ����� ������ � ������� ���� (������������� ����)
testFile = 'C:\Users\market8\Desktop\GitHub\MATLAB_NN\MNIST_perceptron\files\mnist_test.csv';
testData = csvread(testFile);
% ��������������� 0 - � ����� 10
testData(testData(:,1) == 0,1) = 10;

% ��������� ����� ������ � ������� ���� (�������� ����)
trainFile = 'C:\Users\market8\Desktop\GitHub\MATLAB_NN\MNIST_perceptron\files\mnist_train.csv';
trainData = csvread(trainFile);
% ��������������� 0 - � ����� 10
trainData(trainData(:,1) == 0,1) = 10;

%% ������� ��������

tic

% ���-�� ������� ����������
inodes = 784;
% ���-�� ������� ����� 1-��� ����
hnodes = 150;
% ���-�� �������� ����������
onodes = 10;
% ����� �������� �������� 
lr = 0.3;

% ����� ��� ��������� ������� ����. ����� ������� � �������
sigmIn = (hnodes^(-0.75));
% ����� ��� ��������� ������� ����. ����� ������� � ��������
sigmOut = (onodes^(-0.75));

% ��������� ������� ����� ����� �������� ����������� � ������� ����� (�� ������)
wih = sigmIn.*randn(hnodes, inodes);
% ��������� ������� ����� ����� ������� ����� � ��������� ����������� (�� ������)
who = sigmOut.*randn(onodes, hnodes);

% ������� ��������� ����
for i = 1:size(trainData,1)
    aim = ones(onodes,1)*0.01;
    aim(trainData(i,1)) = 0.99;
    inputs = ((trainData(i,2:end)/255 * 0.99) + 0.01)';
    [wih, who] = trainNeural(inputs, aim, wih, who, lr);
end

% ��������� ��������� ����
ans_arr = zeros(size(testData,1),1);
for i = 1:size(testData,1)
    inputsTest = testData(i,2:end)';
    ansNeural = questNeural(inputsTest, wih, who);
    [~,ansMaxInd] = max(ansNeural);
    if testData(i,1) == ansMaxInd
        ans_arr(i) = 1;
    end
end

disp(['������������� ���� �����: ', num2str(sum(ans_arr)/length(ans_arr))])

toc
% %% ���������� ��������� ���� ��� ������ ��������� ��������
% 
% tic
% 
% % ��������� ��������� ����
% inputsTest = imread('E:\Yuriy\MatLab\NBP_algorithm\MatLab\MyNeural\test.png');
% 
% testImg = inputsTest(:,:,1);
% testImg = 255.0 - testImg;
% testImg = im2double(testImg);
% % ���������� �� ������
% imshow(reshape(testImg, 28,28));
% % ������������ ������, ��� ����������� �������������, ��� � ��������
% testImg = rot90(fliplr(testImg),1);
% % ��������������� � ������
% testImg = reshape(testImg, 1, 28*28)';
% 
% ansNeural = questNeural(testImg, wih, who) %#ok
% [~, ansBest] = max(ansNeural); 
% 
% disp([num2str(max(ansNeural)*100), ' - ��������� ����������� ���� ���: ', num2str(ansBest)])
% 
% toc
% 
% %% ������� ���� ������� �� �������
% 
% imgCh = trainData(25,2:end);
% % imgCh = reshape(testImg, 1, 28*28)';
% % imgCh = rot90(imgCh,-1)';
% imshow(reshape(imgCh,28,28));


