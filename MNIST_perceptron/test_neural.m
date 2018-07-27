
clearvars
load('t.mat')

%% параметры

% кол-во входных параметров
inodes = 9;
% кол-во скрытых узлов 1-ого сло€
hnodes = 9;
% кол-во выходных параметров
onodes = 1;
% коэф скорости обучени€ 
lr = 0.1;
% кол-во эпох
epoch = 100;
% тестова€ выборка
t = 0.4;

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

spl = randperm(numel(target))';

% тестовые данные
test_train = data(spl(1:(round(numel(target)*t))),:);
test_target = target(spl(1:(round(numel(target)*t))),:);

% тренировочные данные
train = data(spl(round(numel(target)*t+1):end),:);
train_target = target(spl(round(numel(target)*t+1):end),:);

%% обучаем нейронную сеть

for e = 1:epoch
    % обучаем нейронную сеть
    for i = 1:size(train,1)
        aim = train_target(i);
        inputs = train(i,:)';
        [wih, who] = trainNeural(inputs, aim, wih, who, lr);
    end
    
    % тестируем тренировочные данные
    ans_train = zeros(size(test_train,1),1);
    for i = 1:size(train,1)
        inputsTest = train(i,:)';
        ansNeural = questNeural(inputsTest, wih, who);
        ans_train(i) = ansNeural;
    end
    ans_train = round(ans_train);
    efTrain(e,1) = sum(ans_train==train_target) / size(train_target,1);
    
    % тестируем тестовые данные
    ans_test = zeros(size(test_train,1),1);
    for i = 1:size(test_train,1)
        inputsTest = test_train(i,:)';
        ansNeural = questNeural(inputsTest, wih, who);
        ans_test(i) = ansNeural;
    end
    ans_test = round(ans_test);
    efTest(e,1) = sum(ans_test==test_target)/size(test_target,1);
end
%%
hold on
    plot(efTrain, 'g')
    plot(efTest, 'r')
    % заголовок графика
    title('Titanic neural network')
    % сетка
    grid on
    % подпись по x
    xlabel('epoch')
    % подпись по y
    ylabel('predict')
    % присваевание оси черного цвета
%     set(gca,'color','k')
    % присваевание сетке белого цвета
    set(gca,'gridcolor', 'w')
    % легенда
    legend('training', 'testing')
hold off

%% тестируем нейронную сеть

% тестируем тренировочные данные
ans_train = zeros(size(test_train,1),1);
for i = 1:size(train,1)
    inputsTest = train(i,:)';
    ansNeural = questNeural(inputsTest, wih, who);
    ans_train(i) = ansNeural;
end
ans_train_p = ans_train;
ans_train = round(ans_train);
disp(['Ёффективность сети на тренировочных: ', ...
    num2str(sum(ans_train==train_target)/size(train_target,1))])

% тестируем тестовые данные
ans_test = zeros(size(test_train,1),1);
for i = 1:size(test_train,1)
    inputsTest = test_train(i,:)';
    ansNeural = questNeural(inputsTest, wih, who);
    ans_test(i) = ansNeural;
end
ans_test_p = ans_test;
ans_test = round(ans_test);
disp(['Ёффективность сети на тестовых: ', ...
    num2str(sum(ans_test==test_target)/size(test_target,1))])
    
%% решаем задачу

answer = zeros(size(test,1),1);
for i = 1:size(test,1)
    inputsTest = test(i,:)';
    ansNeural = questNeural(inputsTest, wih, who);
    answer(i) = ansNeural;
end
kaggle = round(answer);

