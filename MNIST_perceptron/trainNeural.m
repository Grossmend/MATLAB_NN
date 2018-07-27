
function [wih, who] = trainNeural(inputs, aim, wih, who, lr)

% ������� �������� ��������� ����

% inputs - ���������� ����� ��������� ����
% aim - ��������� ���� ��������� ���� (� ���� ������ ����������)

% �������� ������� ��� �������� ����
hidden_inputs = wih*inputs;
% ��������� ������� ��� �������� ����
hidden_outputs = logsig(hidden_inputs);

% �������� ������� ��� ��������� ����
funal_inputs = who*hidden_outputs;
% ��������� ������� ��� ��������� ����
final_outputs = logsig(funal_inputs);

% ������ ��������� ����
output_errors = aim - final_outputs;
% ������ �������� ����, �������������� ��������������� ����� ������
hidden_errors = who'*output_errors;

% ���������� ������� ������������� ����� ������� ����� � ��������� �����������
who = who + lr .* (output_errors .* final_outputs .* (1 - final_outputs)) * hidden_outputs';
 
% ���������� ������� ������������� ����� �������� ����������� � ������� �����
wih = wih + lr .* (hidden_errors .* hidden_outputs .* (1 - hidden_outputs)) * inputs';

end

