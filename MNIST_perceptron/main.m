
%% загружаем данные

% you can download this file from https://yadi.sk/d/-oLo6E7Y3ZfJJj
url = 'C:\Users\Grossmend\Desktop\rep\_data\MATLAB\mnist\mnist_data.csv';
data = get_data_mnist(url);

%% разбиваем выборку на тестовую и тренировочную

trainPercent = 0.7;

[trainData, ...
 trainTarget, ...
 testData, ...
 testTarget] = split_mnist_data(data, trainPercent);

%% параметры нейронной сети

% кол-во входных параметров
inodes = size(trainData,2);

% кол-во скрытых узлов 1-ого сло€
hnodes = 150;

% кол-во выходных параметров
onodes = 10;

% коэф. скорости обучени€ 
lr = 0.3;

%% инициализируем начальные матрицы весов

% сигма дл€ начальных весовых коэф. между входным и скрытым
sigmIn = (hnodes^(-0.5));
% сигма дл€ начальных весовых коэф. между скрытым и выходным
sigmOut = (onodes^(-0.5));

% начальна€ матрица весов между входными параметрами и скрытым слоем
wih = sigmIn.*randn(hnodes, inodes);
% начальна€ матрица весов между скрытым слоем и выходными параметрами
who = sigmOut.*randn(onodes, hnodes);

%% главный алгоритм

tic

% обучаем нейронную сеть
for i = 1:size(trainData,1)
    aim = ones(onodes,1)*0.01;
    aim(trainTarget(i)) = 0.99;
    inputs = ((trainData(i,:)/255 * 0.99) + 0.01)';
    [wih, who] = trainNeural(inputs, aim, wih, who, lr);
end

% тестируем нейронную сеть
ans_arr = zeros(size(testData,1),1);
for i = 1:size(testData,1)
    inputsTest = testData(i,:)';
    ansNeural = questNeural(inputsTest, wih, who);
    [~,ansMaxInd] = max(ansNeural);
    if testTarget(i) == ansMaxInd
        ans_arr(i) = 1;
    end
end

% выводим результат нейронной сети
disp(['Ёффективность сети равна: ', num2str(sum(ans_arr)/length(ans_arr))])

toc

%% тестировка нейронной сети дл€ одного параметра входного (png 28x28)

tic

% тестируем нейронную сеть
png = imread('C:\Users\Grossmend\Desktop\rep\_data\MATLAB\mnist\test.png');
testImg = png(:,:,1);
testImg = 255.0 - testImg;

% отображаем на грфике
imshow(reshape(testImg, 28,28));

% приводим картинку к вектору обучающих данных
testImg = reshape(rot90(flip(testImg, 2),1), 1, 28*28)';

% опрашиваем сеть
ansNeural = questNeural(double(testImg), wih, who);

% ищем максимальный результат
[~, ansBest] = max(ansNeural); 

% выводим результат
disp([num2str(max(ansNeural)*100), ' - процентов веро€тности того что: ', num2str(ansBest)])

toc

%%