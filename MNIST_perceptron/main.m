
%% ��������� ������

% you can download this file from https://yadi.sk/d/-oLo6E7Y3ZfJJj
url = 'C:\Users\Grossmend\Desktop\rep\_data\MATLAB\mnist\mnist_data.csv';
data = get_data_mnist(url);

%% ��������� ������� �� �������� � �������������

trainPercent = 0.7;

[trainData, ...
 trainTarget, ...
 testData, ...
 testTarget] = split_mnist_data(data, trainPercent);

%% ��������� ��������� ����

% ���-�� ������� ����������
inodes = size(trainData,2);

% ���-�� ������� ����� 1-��� ����
hnodes = 150;

% ���-�� �������� ����������
onodes = 10;

% ����. �������� �������� 
lr = 0.3;

%% �������������� ��������� ������� �����

% ����� ��� ��������� ������� ����. ����� ������� � �������
sigmIn = (hnodes^(-0.5));
% ����� ��� ��������� ������� ����. ����� ������� � ��������
sigmOut = (onodes^(-0.5));

% ��������� ������� ����� ����� �������� ����������� � ������� �����
wih = sigmIn.*randn(hnodes, inodes);
% ��������� ������� ����� ����� ������� ����� � ��������� �����������
who = sigmOut.*randn(onodes, hnodes);

%% ������� ��������

tic

% ������� ��������� ����
for i = 1:size(trainData,1)
    aim = ones(onodes,1)*0.01;
    aim(trainTarget(i)) = 0.99;
    inputs = ((trainData(i,:)/255 * 0.99) + 0.01)';
    [wih, who] = trainNeural(inputs, aim, wih, who, lr);
end

% ��������� ��������� ����
ans_arr = zeros(size(testData,1),1);
for i = 1:size(testData,1)
    inputsTest = testData(i,:)';
    ansNeural = questNeural(inputsTest, wih, who);
    [~,ansMaxInd] = max(ansNeural);
    if testTarget(i) == ansMaxInd
        ans_arr(i) = 1;
    end
end

% ������� ��������� ��������� ����
disp(['������������� ���� �����: ', num2str(sum(ans_arr)/length(ans_arr))])

toc

%% ���������� ��������� ���� ��� ������ ��������� �������� (png 28x28)

tic

% ��������� ��������� ����
png = imread('C:\Users\Grossmend\Desktop\rep\_data\MATLAB\mnist\test.png');
testImg = png(:,:,1);
testImg = 255.0 - testImg;

% ���������� �� ������
imshow(reshape(testImg, 28,28));

% �������� �������� � ������� ��������� ������
testImg = reshape(rot90(flip(testImg, 2),1), 1, 28*28)';

% ���������� ����
ansNeural = questNeural(double(testImg), wih, who);

% ���� ������������ ���������
[~, ansBest] = max(ansNeural); 

% ������� ���������
disp([num2str(max(ansNeural)*100), ' - ��������� ����������� ���� ���: ', num2str(ansBest)])

toc

%%