# Bibliotecas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\bianca\\Downloads\\dataset_SIN492.csv')

X = data.drop('target', axis=1)
y = data['target']

# Manipulação de dados ausentes
X = X.fillna(X.mean())

# Oversampling
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

# Z-score
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_resampled)

# Dividindo em treinamento/teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_resampled, test_size=0.2, random_state=42)

# Criando o modelo 
modelo_random_forest = RandomForestClassifier(random_state=42)

# Treinando
modelo_random_forest.fit(X_train, y_train)

# Previsões - teste
previsoes_rf = modelo_random_forest.predict(X_test)

# Acurácia
acuracia_rf = accuracy_score(y_test, previsoes_rf)
print(f'Acurácia do modelo Random Forest: {acuracia_rf}')

# Relatório
relatorio_classificacao_rf = classification_report(y_test, previsoes_rf)
print(f'Relatório de Classificação do Random Forest:\n{relatorio_classificacao_rf}')

# Matriz de Confusão
matriz_confusao = confusion_matrix(y_test, previsoes_rf)
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão do Random Forest')
plt.show()

# Cross-validation
scores = cross_val_score(modelo_random_forest, X_normalized, y_resampled, cv=5)
print(f'Acurácias em cada fold: {scores}')
print(f'Acurácia média da validação cruzada: {scores.mean()}')

# Hiperparâmetros
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_normalized, y_resampled)
print("Melhores parâmetros:", grid_search.best_params_)

# Gráfico de Importância de Características
feature_importances = modelo_random_forest.feature_importances_
feature_names = X.columns
plt.barh(feature_names, feature_importances)
plt.xlabel('Importância')
plt.ylabel('Características')
plt.title('Importância das Características no Modelo Random Forest')
plt.show()
