from helpers.help import handleData, interpretation
from network.network import RNN
import torch
import torch.nn as nn
import torch.optim as optim

# data = [(train, expected), (train, expected), ...]
data = handleData('data/dataset.txt', 'data/expecteds.txt')

""" 
#Visualizar os dados:
for item in data:
    print(f'real: {"".join(chr(i) for i in item[0])} // expected {item[1]}') 
"""

#instancia a rede neural
net = RNN(60, 54, 1, 60)
lr = 0.5
gradienteErro = nn.MSELoss()
correcaoErro = optim.Adadelta(net.parameters(), lr=lr)
epochs = 100
trainLen = int(len(data)*0.8)

for epoch in range(epochs):
    for i in range(0, trainLen):
        output = net(data[i][0])

        #print(output[0])
        #zera o gradiente de erro
        correcaoErro.zero_grad()

        #calcula o erro
        loss = gradienteErro(output[0], data[i][1])

        #calcula o gradiente de erro
        loss.backward()

        #atualiza os pesos
        correcaoErro.step()
        if (i+1) % 30 == 0:
                print(f'Época [{epoch+1}/{epochs}], Step {i+1}/{len(data)}, Loss: {loss.item()}')

                if loss.item() < 0.12 and lr > 0.1:
                    lr *= 0.85
                elif loss.item() < 0.02 and lr < 0.15 and lr > 0.0007:
                    lr *= 0.75

with torch.no_grad():
    correct = []
    for i in range(trainLen, len(data)):

        #passa os dados de validação pela rede neural
        output = net(data[i][0])

        #interpreta a saída da rede neural
        output = interpretation(output[0])

        #printa os resultados
        """ print(f'Pattern: \n')
        for c in [chr(int(i)) for i in data[i][0]]:
            print(c, end='') """
        print(f'\nExpected: {data[i][1]} \n Output: {output}')
        #print(f'Loss: {gradienteErro(output[0], data[i][1]).item()}')
        print('---')
        succ = 0
        #calcula a acurácia
        for item in range(0, len(output)):
            if output[item] == data[i][1][item]:
                succ += 1
        correct.append(succ/output.size()[0])
        print(correct[-1])
    total = 0

    for item in correct:
        total += item
    
    print('-=-=-=-=-=-=-=-=-=-')
    print(total)
    print(f'{len(data)-trainLen} = {len(data)} - {trainLen}')
    print(f'Acurácia: {(100 * total)/(len(data)-trainLen)}%')