
clearvars

url = 'C:\Users\Grossmend\Desktop\repositories\MATLAB_NN\files\mnist_test.csv';
data = get_data_mnist(url);

function [data] = get_data_mnist(url)
    
    % ������� ������� ����
    try
        data = csvread(url);
    catch ME
        if (strcmp(ME.identifier,'MATLAB:csvread:FileNotFound'))
            error(['can not find file ', url]);
        end
    end
    
    % ���� ���� ����������, �� ������� csvread �� ������� ���
    if exist('data', 'var') == 0
        error('problem read file');
    end
    
    % ��������� ����� � ����� ���� ����� 2 ��������� ������
    if size(data,1) < 2
        error('length data must be more then 2')
    end
    
end