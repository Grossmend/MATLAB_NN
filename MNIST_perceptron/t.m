
%% параметры

load('t.mat')

% кол-во входных параметров
inodes = 8;
% кол-во скрытых узлов 1-ого сло€
hnodes = 50;
% кол-во выходных параметров
onodes = 1;
% коэф скорости обучени€ 
lr = 0.3;
% кол-во эпох
epoch = 20;
% тестова€ выборка
t = 0.15;

%% главный алгоритм

% сигма дл€ начальных весовых коэф. между входным и скрытым
sigmIn = (hnodes^(-0.5));
% сигма дл€ начальных весовых коэф. между скрытым и выходным
sigmOut = (onodes^(-0.5));

% матрица весов между входными параметрами и скрытым слоем (по √ауссу)
wih = sigmIn.*randn(hnodes, inodes);
% матрица весов между скрытым слоем и выходными параметрами (по √ауссу)
who = sigmOut.*randn(onodes, hnodes);

%% разбиваем выборку на тренировочную и тестовую

spl = randperm(numel(target));

% тестовые данные
test = data(spl(1:(round(numel(target)*t))),:);
test_target = target(spl(1:(round(numel(target)*t))),:);

% тренировочные данные
train = data(spl(round(numel(target)*t+1):end),:);
train_target = target(spl(round(numel(target)*t+1):end),:);

%% обучаем нейронную сеть
for e = 1:epoch
    for i = 1:size(train,1)
        aim = train_target(i);
        inputs = train(i,:)';
        % обучаем нейронную сеть
        [wih, who] = trainNeural(inputs, aim, wih, who, lr);
    end
end

%% тестируем нейронную сеть

% тестируем тренировочные данные
ans_train = zeros(size(test,1),1);
for i = 1:size(train,1)
    inputsTest = train(i,:)';
    ansNeural = questNeural(inputsTest, wih, who);
    ans_train(i) = ansNeural;
end
ans_train = round(ans_train);
disp(['Ёффективность сети на тренировочных: ', ...
    num2str(sum(ans_train==train_target)/size(train_target,1))])

% тестируем тестовые данные
ans_test = zeros(size(test,1),1);
for i = 1:size(test,1)
    inputsTest = test(i,:)';
    ansNeural = questNeural(inputsTest, wih, who);
    ans_test(i) = ansNeural;
end

ans_test = round(ans_test);

disp(['Ёффективность сети на тестовых: ', ...
    num2str(sum(ans_test==test_target)/size(test_target,1))])