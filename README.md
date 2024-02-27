# LSTM-TextRecognition
Este desafio foi proposto como solução no curso AI900 da Microsoft em parseria com a DIO.

# Tecnologias Utilizadas
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pytorch](https://img.shields.io/badge/python-gray?style=for-the-badge&logo=pytorch&logoColor=red)

## tree do projeto
| Pasta   | Conteúdo   |
|--------|------------|
|network| Contém toda a nossa rede neural|
|data| Contem o nosso dataset|
|helpers| Contém funções que lidam com o dataset e também que interagem com a interpretação de dados da saída da rede neural|

# Passos do Projeto
A idéia para este projeto veio de um comentário que um amigo me fez. Ele trabalha com transformers, retirando informações de documentos escrito por pessoas. Estes documentos não possuem um rigido padrão, e algumas pessoas faziam "enfeites" no texto para chamar a atenção de outras pessoas.

Entretanto, quando passado este conteúdo para um transformers, a rede não conseguia entender o que exatamente estava tentando ser escrito ali, e apresentava resultados inconsistentes.

Este projeto que submeto como parte do curso AI900 da Microsoft foi criado com base nesta experiência.

Neste projeto, temos um dataset de [textos com "enfeites"](https://github.com/JoaoLucasMoraisOrtiz/LSTM-TextRecognition/blob/main/data/dataset.txt). O objetivo é reconhecer o texto que existe antes deste padrão de enfeite, dentro do padrão e depois do padrão, mas eliminando o padrão. Assim a resposta da rede deve ser um vetor com o mesmo tamanho do vetor de entrada com 0 nos caracteres que devem ser ignorados, e 1 nos caracteres que devem ser considerados.
Posteriormente podemos múltiplicar caractere por caractere do texto original e teremos apenas informações relevantes, e não o "enfeite".

Este dataset não alcansou um bom nível de desempenho, tendo ficado entre 75%-80% de precisão.
Talvez se eu utilizasse transformers ao invés de uma rede LSTM eu tivesse um melhor resultado.