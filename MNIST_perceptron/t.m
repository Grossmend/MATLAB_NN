
%% ���������

load('t.mat')

% ���-�� ������� ����������
inodes = 8;
% ���-�� ������� ����� 1-��� ����
hnodes = 50;
% ���-�� �������� ����������
onodes = 1;
% ���� �������� �������� 
lr = 0.3;
% ���-�� ����
epoch = 20;
% �������� �������
t = 0.15;

%% ������� ��������

% ����� ��� ��������� ������� ����. ����� ������� � �������
sigmIn = (hnodes^(-0.5));
% ����� ��� ��������� ������� ����. ����� ������� � ��������
sigmOut = (onodes^(-0.5));

% ������� ����� ����� �������� ����������� � ������� ����� (�� ������)
wih = sigmIn.*randn(hnodes, inodes);
% ������� ����� ����� ������� ����� � ��������� ����������� (�� ������)
who = sigmOut.*randn(onodes, hnodes);

%% ��������� ������� �� ������������� � ��������

spl = randperm(numel(target));

% �������� ������
test = data(spl(1:(round(numel(target)*t))),:);
test_target = target(spl(1:(round(numel(target)*t))),:);

% ������������� ������
train = data(spl(round(numel(target)*t+1):end),:);
train_target = target(spl(round(numel(target)*t+1):end),:);

%% ������� ��������� ����
for e = 1:epoch
    for i = 1:size(train,1)
        aim = train_target(i);
        inputs = train(i,:)';
        % ������� ��������� ����
        [wih, who] = trainNeural(inputs, aim, wih, who, lr);
    end
end

%% ��������� ��������� ����

% ��������� ������������� ������
ans_train = zeros(size(test,1),1);
for i = 1:size(train,1)
    inputsTest = train(i,:)';
    ansNeural = questNeural(inputsTest, wih, who);
    ans_train(i) = ansNeural;
end
ans_train = round(ans_train);
disp(['������������� ���� �� �������������: ', ...
    num2str(sum(ans_train==train_target)/size(train_target,1))])

% ��������� �������� ������
ans_test = zeros(size(test,1),1);
for i = 1:size(test,1)
    inputsTest = test(i,:)';
    ansNeural = questNeural(inputsTest, wih, who);
    ans_test(i) = ansNeural;
end

ans_test = round(ans_test);

disp(['������������� ���� �� ��������: ', ...
    num2str(sum(ans_test==test_target)/size(test_target,1))])