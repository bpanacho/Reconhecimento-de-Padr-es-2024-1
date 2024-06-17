# Bibliotecas
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
caminho_arquivo_parquet = 'C:\\Users\\Bianca\\Downloads\\dataset_SIN492.parquet'
dados_parquet = pd.read_parquet(caminho_arquivo_parquet)

# Separar as colunas de características e a coluna alvo
colunas_features = [f'feature{i}' for i in range(16)]
coluna_alvo = 'target'
X = dados_parquet[colunas_features]
y = dados_parquet[coluna_alvo]

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados usando Z-score
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir os hiperparâmetros a serem testados
parametros = {
    'hidden_layer_sizes': [(128, 64)],
    'alpha': [0.0001],
    'learning_rate': ['constant'],
    'max_iter': [1000]
}

# Criar o modelo de rede neural
modelo_mlp = MLPClassifier(random_state=42)

# Realizar a busca em grade
grid_search = GridSearchCV(modelo_mlp, parametros, cv=3)
grid_search.fit(X_train, y_train)

# Imprimir os melhores hiperparâmetros encontrados
print("Melhores Hiperparâmetros MLPClassifier:", grid_search.best_params_)

# Obter o modelo MLP com os melhores hiperparâmetros
modelo_mlp_final = grid_search.best_estimator_

# Criar o ensemble de modelos
ensemble = VotingClassifier(estimators=[
    ('mlp', modelo_mlp_final),
], voting='soft')  # 'soft' para média ponderada

# Treinar o ensemble com os dados de treinamento
ensemble.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
previsoes_ensemble = ensemble.predict(X_test)

# Avaliar a acurácia do ensemble
previsoes_ensemble = ensemble.predict(X_test)
acuracia_ensemble = accuracy_score(y_test, previsoes_ensemble)
print(f'Acurácia do Ensemble: {acuracia_ensemble}')

# Criar o modelo MLP com os melhores hiperparâmetros
modelo_mlp_final = grid_search.best_estimator_

# Fazer previsões no conjunto de teste com a MLP
previsoes_mlp = modelo_mlp_final.predict(X_test)

# Calcular a matriz de confusão para a MLP
matriz_confusao_mlp = confusion_matrix(y_test, previsoes_mlp)

# Calcular as porcentagens de acertos para cada classe
porcentagens_acertos_mlp = matriz_confusao_mlp / matriz_confusao_mlp.sum(axis=1)[:, np.newaxis]

# Configurar o Seaborn para gráficos mais bonitos
sns.set()

# Plotar a matriz de confusão da MLP como um heatmap
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(porcentagens_acertos_mlp, annot=True, fmt=".2%", cmap="Blues", cbar=False, xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title("Mapa de Calor das Porcentagens de Acertos (MLP)")
plt.xlabel("Previsão")
plt.ylabel("Real")

# Plotar a curva ROC da MLP
probabilidades_mlp = modelo_mlp_final.predict_proba(X_test)[:, 1]
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, probabilidades_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

plt.subplot(1, 2, 2)
plt.plot(fpr_mlp, tpr_mlp, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_mlp:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC (MLP)')
plt.legend(loc="lower right")

plt.show()
