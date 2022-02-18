# Deep-learning
Foi utilizado o banco de dados FER - 13, o qual pode ser encontrado no site “https://www.kaggle.com/msambare/fer2013”.
Apesar de o dataset possuir amostras para sete tipos de expressões faciais, isto é, “raiva, nojo, medo, feliz, neutro, triste e surpreso”, apenas três delas foram utilizadas, a saber “feliz, neutro e triste”.
A quantidade de amostras por expressão facial estava desequilibrada, e com isso a pasta que possuía muito mais amostras que as demais teve algumas imagens excluídas.
No total, foram aproximadamente 5000 amostras de treinamento e 1200 amostras de teste para cada uma das três expressões faciais analisadas.
O desempenho é analisado através da matriz de confusão.
Após 80 épocas, o valor de acurácia obtido foi 0.8857 e a acurácia de validação foi de 0.7597.
