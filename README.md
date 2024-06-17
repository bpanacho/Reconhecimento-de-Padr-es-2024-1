# Reconhecimento de Padrões 2024-1
### Autores: Bianca Panacho Ferreira e Pedro Henrique Campos Moreira
Projeto da Disciplina de Verão Reconhecimento de Padrões da Universidade Federal de Viçosa 

# Avaliando o Desempenho do Random Forest e Redes Neurais em Problemas de Classificação Binária
### Tecnologias Utilizadas
<table>
  <tr>
    <td>LINGUAGEM</td>
    <td>BIBLIOTECAS</td>
    <td>AMBIENTE DE DESENVOLVIMENTO </td>
  </tr>
  <tr>
    <td>Python</td>
    <td>Pandas, Scikit-Learn, Numpy, Seaborn e Matplotlib</td>
    <td>Visual Studio Code e Jupyter Notebook</td>
    
  </tr>
</table>


### Modelo Escolhido
+ Random Forest.
+ É um algoritmo de aprendizado de máquina que cria múltiplas árvores de decisão e combina suas previsões para melhorar a precisão na classificação. Cada árvore "vota" na classe, e a classe mais votada se torna a predição final.
+ Exemplo: classificação de e-mails como spam ou não spam.
+ Acurácia alcançada: 74%.

*Optamos pela Random Forest devido à sua interpretabilidade, robustez a hiperparâmetros, capacidade de lidar de forma eficaz com dados pequenos ou ruídosos, treinamento mais eficiente, menor exigência de normalização de dados e resistência ao overfitting.*

### Rede Neural
+ Rede Neural MLP.
+ Uma Rede Neural MLP (Multilayer Perceptron) é um modelo de aprendizado profundo que consiste em camadas de neurônios conectados. Usada para classificação, ela aprende padrões complexos nos dados durante o treinamento, sendo eficaz em problemas como reconhecimento de imagem, previsões financeiras, etc.
+ Exemplo: classificação de dígitos escritos à mão em um conjunto de dados MNIST.
+ Acurácia alcançada: 72%.


# Execução do projeto:

1) Dê um git clone para executar o projeto.

2) Instale as bibliotecas necessárias.
```
pip install pandas scikit-learn numpy seaborn matplotlib
```

3) Execute o Código:

Abra o arquivo que contém o código da Rede Neural (MLP):
- Execute o script utilizando um ambiente Python.
- Algumas aplicações podem exigir dados específicos ou configurações adicionais.
- Certifique-se de entender os requisitos específicos de cada aplicação.

Abra o arquivo que contém o código da Random Forest:
- Execute o script utilizando um ambiente Python.
- Antes de executar, verifique se você possui um conjunto de dados disponível ou ajuste o código para carregar seus próprios dados.
- Certifique-se de compreender os hiperparâmetros definidos no código e ajuste conforme necessário, dependendo do seu problema específico.
- Avalie a necessidade de ajustar outros parâmetros, como a estrutura da rede neural, de acordo com a complexidade do seu problema.


## Explicação da estrutura e estratégias aplicadas ao código:

### Como implementar a Rede Neural MLP:

Passo 1: Importar Bibliotecas e Carregar Dados

Importe as seguintes bibliotecas e carregue seus dados a partir de um arquivo, por exemplo, pd.read_parquet.
- pandas
- scikit-learn
- numpy
- seaborn
- matplotlib
  
Passo 2: Pré-processamento dos Dados 

- Separe suas características (features) e a coluna alvo do conjunto de dados.
- Divida os dados em conjuntos de treinamento e teste usando train_test_split.
- Padronize os dados para garantir que todas as características estejam na mesma escala.

Passo 3: Treinamento da Rede Neural MLP com Busca em Grade 

- Defina os hiperparâmetros que deseja testar para sua MLP, como o tamanho das camadas ocultas e a taxa de aprendizado.
- Use a busca em grade (GridSearchCV) para encontrar os melhores hiperparâmetros com base no desempenho nos dados de treinamento.

Passo 4: Ensemble de Modelos com Votação Suave 

- Crie um ensemble usando o método VotingClassifier.
- Adicione o modelo MLP com os melhores hiperparâmetros ao ensemble.
- Treine o ensemble com os dados de treinamento.

Passo 5: Avaliação do Ensemble 📊✅

- Faça previsões no conjunto de teste usando o ensemble e avalie sua acurácia usando métricas como accuracy_score.

Passo 6: Visualização de Resultados (Opcional) 📈🔍

- Crie uma matriz de confusão para a MLP ou uma curva ROC para avaliar o desempenho do modelo.

### Como implementar a Random Forest:

Passo 1: Importar Bibliotecas e Carregar Dados.
 
Importe as seguintes bibliotecas e carregue seus dados a partir de um arquivo, por exemplo, pd.read_parquet.
- pandas
- scikit-learn
- numpy
- seaborn
- matplotlib

Passo 2: Pré-processamento dos Dados 

- Separe as características (features) e a coluna alvo do conjunto de dados.
- Divida os dados em conjuntos de treinamento e teste usando train_test_split.
- Realize qualquer pré-processamento necessário.

Passo 3: Treinamento da Random Forest 🚀

- Crie um modelo de Random Forest usando RandomForestClassifier.
- Ajuste o modelo aos dados de treinamento para permitir que ele aprenda padrões no conjunto de treinamento.

Passo 4: Avaliação do Modelo 

- Faça previsões no conjunto de teste e avalie o desempenho da Random Forest usando métricas como acurácia, matriz de confusão e curva ROC.

Passo 5: Visualização dos Resultados (Opcional) 📈

- Crie a visualização da importância das características ou uma representação gráfica da árvore de decisão.

# Visualização dos Resultados 


## AD - Random Forest
+ Representação visual da floresta aleatória gerada:

![Figure_1](https://github.com/JFcamp/reconhecimento-de-padr-es-/assets/149902237/4f645ea8-3da7-4057-b5d5-9424e029d2c2)

## Rede Neural
+ Curva ROC - Comparação do modelo com a linha de "Chute":
  
![Captura de tela 2024-01-25 171428](https://github.com/JFcamp/reconhecimento-de-padr-es-/assets/149902237/2befd8fe-4e88-49ed-9dee-2fa54a0aefb4)


