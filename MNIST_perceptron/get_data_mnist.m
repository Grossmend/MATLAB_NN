
function [data] = get_data_mnist(url)

% fuction get data mnsit from csv. You can download this file from 
% https://yadi.sk/d/-oLo6E7Y3ZfJJj
% Grossmend, 2018

% пробуем открыть файл
try
    data = csvread(url);
catch ME
    if (strcmp(ME.identifier,'MATLAB:csvread:FileNotFound'))
        error(['can not find file ', url]);
    end
end

% если файл существует, но функция csvread не создала его
if exist('data', 'var') == 0
    error('problem read file');
end

% проверяем чтобы в файле было более 2 обучаемых данных
if size(data,1) < 2
    error('length data must be more then 2')
end

end