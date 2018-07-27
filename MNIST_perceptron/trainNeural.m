
function [wih, who] = trainNeural(inputs, aim, wih, who, lr)

% функция обучения неройнной сети

% inputs - параматеры входа нейронной сети
% aim - параметры цели нейронной сети (к чему должны стремиться)

% входящие сигналы для скрытыго слоя
hidden_inputs = wih*inputs;
% выходящие сигналы для скрытого слоя
hidden_outputs = logsig(hidden_inputs);

% входящие сигналы для выходного слоя
funal_inputs = who*hidden_outputs;
% исходящие сигналы для выходного слоя
final_outputs = logsig(funal_inputs);

% ошибки выходного слоя
output_errors = aim - final_outputs;
% ошибки скрытого слоя, распределенные пропорционально весам связей
hidden_errors = who'*output_errors;

% обновление весовых коэффициентов между скрытым слоем и выходными параметрами
who = who + lr .* (output_errors .* final_outputs .* (1 - final_outputs)) * hidden_outputs';
 
% обновление весовых коэффициентов между входными параметрами и скрытым слоем
wih = wih + lr .* (hidden_errors .* hidden_outputs .* (1 - hidden_outputs)) * inputs';

end

