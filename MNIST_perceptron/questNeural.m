
function ansNeural = questNeural(inputs, wih, who)

% функция тестирования нейронной сети

% inputs - входные параметры нейронной сети
% wih - обученная матрица коэффициентов весов входной-скрытый
% who - обученная матрциа коэффициентов вестов скрытый-выходной

% входящие сигналы для скрытыго слоя
hidden_inputs = wih*inputs;
% исходящие сигналы для скрытого слоя
hidden_outputs = logsig(hidden_inputs);

% входящие сигналы для выходного слоя
funal_inputs = who*hidden_outputs;
% исходящие сигналы для выходного слоя
ansNeural = logsig(funal_inputs);

end
