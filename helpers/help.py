from random import randint, shuffle
import math
from pathlib import Path
import torch
def handleData(file_path, answer_patch):
    """ 
        Função que lê os dados de nosso arquivo dataset, e os transforma em vetores de inteiros
        :param file_path: o caminho do arquivo que contém os dados
        :type file_path: str
        :param answer_patch: o caminho do arquivo que contém as respostas esperadas
        :type answer_patch: str
        :return: uma lista de vetores de inteiros
    """
    
    #variáveis globais
    dataNum = []
    expected = []
    globalPath = str(Path.cwd())+ '/'
    try:
        # Lê o arquivo do dataset e separa os dados em uma lista
        with open(globalPath + file_path, 'r') as file:
            dataText = file.read().split(';')
            for i in dataText[:-1]:
                dataNum.append([ord(item) for item in i[:-1]])
    except FileNotFoundError:
        print(f"O arquivo {globalPath + file_path} não foi encontrado.")
        exit()
    
    try:
        # Lê o arquivo de respostas e separa os dados em uma lista
        with open(globalPath + answer_patch, 'r') as file:
            e = file.read().split(';')
            for i in e[:-1]:
                expected.append([int(c) for c in i.split(' ')[:-1]])
    except FileNotFoundError:
        print(f"O arquivo {answer_patch} não foi encontrado.")
        exit()

    for item in range(0, len(dataNum)):

        # adciona caracteres ASCII aleatórios.
        if(len(dataNum[item]) < 60):
            
            #calcula o número de caracteres que serão colocados
            f = math.ceil( (60-len(dataNum[item]))/2 )

            #adciona caracteres aleatórios no começo e no final do vetor
            dataNum[item] = [randint(65, 90) for _ in range(0, f )] + dataNum[item] + [randint(97, 122) for _ in range(0, f - 1)]
            
            #coloca os caracteres randômicos como esperados
            expected[item] = [1 for _ in range(0, f )] + expected[item] + [1 for _ in range(0, f - 1)]
        else:
            pass

        #adciona espaços no final do vetor de treinamento
        if(len(dataNum[item]) < 60):
            while (len(dataNum[item]) < 60):
                dataNum[item] = dataNum[item] + [32]
        
        

        #adciona espaços no final do vetor de resṕsta
        if(len(expected[item]) < 60):
            while (len(expected[item]) < 60):
                expected[item] = expected[item] + [1]

    #transforma os vetores em tensores
    dataNum = torch.tensor(dataNum, dtype=torch.float32)
    expected = torch.tensor(expected, dtype=torch.float32)

    fData = []
    # faz um zip entre os vetores de treinamento e os vetores de resposta
    for i in range(0, len(dataNum)):
        fData.append([dataNum[i], expected[i]])

    shuffle(fData)
    
    #retorna o vetor de resposta
    return fData

def interpretation(output):
    """ 
        Função que interpreta a saída da rede neural. Não podemos utilizar uma
        função do tipo degrau pois ela não é derivável. Portanto vamos aplica-la agora.
        Basicamente vamos encontrar a mediana da saída da rede neural, e então vamos
        definir o que for acima dela como 1, e abaixo como 0.
        :param output: saída da rede neural
        :type output: torch.tensor
        :return: torch.tensor
    """
    median = torch.median(output)
    for i in range(0, len(output)):
        if output[i] > median:
            output[i] = 1
        else:
            output[i] = 0
    
    return output