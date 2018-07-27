
function ansNeural = questNeural(inputs, wih, who)

% ������� ������������ ��������� ����

% inputs - ������� ��������� ��������� ����
% wih - ��������� ������� ������������� ����� �������-�������
% who - ��������� ������� ������������� ������ �������-��������

% �������� ������� ��� �������� ����
hidden_inputs = wih*inputs;
% ��������� ������� ��� �������� ����
hidden_outputs = logsig(hidden_inputs);

% �������� ������� ��� ��������� ����
funal_inputs = who*hidden_outputs;
% ��������� ������� ��� ��������� ����
ansNeural = logsig(funal_inputs);

end
