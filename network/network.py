import torch
import torch.nn as nn
import torch.nn.functional as functional

#nossa rede neural recorrente
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """ 
            função que inicializa a rede neural recorrente
            @param input_size: tamanho da entrada
            @param hidden_size: tamanho da camada escondida
            @param num_layers: número de camadas
            @param num_classes: número de classes
        """

        #inicializa a nossa rede neural como sendo uma RNN
        super(RNN, self).__init__()

        #tamanho do vetor da nossa memória de curto prazo
        self.hidden_size = hidden_size

        #número de camadas (não número de iterações)
        self.num_layers = 1

        #nossa camada lstm
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        #camada fully connected, utilizada para interpretar a saída da lstm, retornando uma das classes possíveis (dado que estamos trabalhando com uma tarefa de classificação)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
            função que realiza a passagem para frente da rede neural
            @param x: entrada da rede neural
            @return: saída da rede neural
        """
        x = x.view(-1, 1, 60)
        #print(x[0].size())
        #inicializa a memória da lstm hidden state e cell state como 0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        #passa a entrada pela lstm
        out, _ = self.lstm(x, (h0, c0))

        #passa a última saída da lstm para cada entrada Xn do batch pela camada fully connected
        #isto retorna um vetor (batch_size, num_classes) onde cada item é um array com a probabilidade de cada classe
        out = torch.sigmoid_(self.fc(out[:, -1, :]))
        return out