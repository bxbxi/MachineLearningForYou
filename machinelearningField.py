import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
def load_file():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Bir dosya seçin",
        filetypes=[("CSV dosyaları", "*.csv")]
    )

    if not file_path:
        print("Dosya seçilmedi.")
        return None

    try:
        df = pd.read_csv(file_path)
        print("Dosya başarıyla yüklendi!")
        return df
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None


def select_unique_values(df):
    root = tk.Tk()
    root.withdraw()

    categorical_columns = df.select_dtypes(include=['object']).columns

    if not categorical_columns.any():
        print("Kategorik sütun bulunamadı.")
        return None, None

    column = simpledialog.askstring("Sütun Seçimi",
                                    f"Kategori sütunlarından birini seçin: {', '.join(categorical_columns)}")

    if column not in categorical_columns:
        messagebox.showerror("Hata", "Geçersiz sütun adı!")
        return None, None

    unique_values = df[column].unique()

    selected_values = simpledialog.askstring("Değer Seçimi",
                                             f"{column} sütunundaki benzersiz değerler: {', '.join(map(str, unique_values))}\n\nBirden fazla değer seçin (virgül ile ayırın):")

    if selected_values:
        selected_values = [val.strip() for val in selected_values.split(',')]
        return column, selected_values
    else:
        messagebox.showwarning("Uyarı", "Hiçbir değer seçilmedi.")
        return None, None


def build_and_evaluate_models(df, selected_column=None, selected_values=None):
    print("Veri Sütunları: ", df.columns)

    target_column = simpledialog.askstring("Hedef Sütun", "Hedef değişken (target) sütununu giriniz:")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if selected_column and selected_values:
        X = X[X[selected_column].isin(selected_values)]
        y = y[X.index]

    if X.isnull().sum().sum() > 0:
        print("Eksik veriler tespit edildi. Eksik veriler dolduruluyor...")
        X = X.fillna(X.median())

    X = pd.get_dummies(X)

    X = X.select_dtypes(include=[np.number])

    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns].hist(bins=15, figsize=(15, 10), layout=(5, 3))
    plt.suptitle('Sayısal Değişkenlerin Dağılımı')
    plt.show()

    plot_correlation_matrix(df)

def plot_correlation_matrix(df):
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    numerical_df = df[numerical_columns]

    plt.figure(figsize=(12, 8))
    correlation_matrix = numerical_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Korelasyon Matrisi')
    plt.show()

def build_and_evaluate_models(df):
    print("Veri Sütunları: ", df.columns)

    target_column = input("Hedef değişken (target) sütununu giriniz: ")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.isnull().sum().sum() > 0:
        print("Eksik veriler tespit edildi.")
        X = X.fillna(X.median())

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {}
        },
        "Ridge Regression": {
            "model": Ridge(),
            "params": {
                "alpha": [0.1, 1, 10, 100]
            }
        },
        "Lasso Regression": {
            "model": Lasso(),
            "params": {
                "alpha": [0.1, 1, 10, 100]
            }
        },
        "Decision Tree Regression": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest Regression": {
            "model": RandomForestRegressor(n_estimators=100, random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Gradient Boosting Regression": {
            "model": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        },
        "Support Vector Regression": {
            "model": SVR(),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "epsilon": [0.01, 0.1, 0.2]
            }
        },
        "K-Nearest Neighbors Regression": {
            "model": KNeighborsRegressor(),
            "params": {
                "n_neighbors": [3, 5, 7, 10],
                "weights": ["uniform", "distance"]
            }
        },
    }

    results = []
    best_models = {}

    for name, model_info in models.items():
        print(f"\n{name} modelini eğitiyoruz...")
        model = model_info["model"]
        params = model_info["params"]

        if params:
            grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"{name} için en iyi hiperparametreler: {best_params}")
        else:
            best_model = model.fit(X_train, y_train)
            best_params = {}

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_score = cross_val_score(best_model, X, y, cv=5).mean()
        print(f"{name} Ortalama Kare Hatası (MSE): {mse}")
        print(f"{name} R² Skoru: {r2}")
        print(f"{name} 5 Katlı Çapraz Doğrulama Skoru: {cv_score}")
        results.append((name, mse, r2, cv_score))
        joblib.dump(best_model, f"{name.replace(' ', '_').lower()}_model.pkl")
        best_models[name] = best_model

    results_df = pd.DataFrame(results, columns=["Model", "MSE", "R²", "Cross-Validation Score"])

    best_mse_model = results_df.loc[results_df['MSE'].idxmin()]
    best_r2_model = results_df.loc[results_df['R²'].idxmax()]
    best_cv_model = results_df.loc[results_df['Cross-Validation Score'].idxmax()]

    print(f"\nEn düşük MSE skoru olan model: {best_mse_model['Model']} - MSE: {best_mse_model['MSE']}")
    print(f"En yüksek R² skoru olan model: {best_r2_model['Model']} - R²: {best_r2_model['R²']}")
    print(f"En yüksek Çapraz Doğrulama skoru olan model: {best_cv_model['Model']} - CV Score: {best_cv_model['Cross-Validation Score']}")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Model", y="Cross-Validation Score", palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title("Modellerin Çapraz Doğrulama Skorları")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Model", y="MSE", palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title("Modellerin Ortalama Kare Hatası (MSE)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Model", y="R²", palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title("Modellerin R² Skorları")
    plt.tight_layout()
    plt.show()

    def plot_predictions(y_test, y_pred, model_name):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
        plt.title(f'{model_name} Tahmin Sonuçları')
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahmin Değerleri')
        plt.show()

def load_and_predict(model_name, file_path):
    try:
        model = joblib.load(f"{model_name.replace(' ', '_').lower()}_model.pkl")
    except FileNotFoundError:
        print(f"{model_name} modeli bulunamadı. Önce modeli eğitmeniz gerekiyor.")
        return

    df = pd.read_csv(file_path)
    print("Dosya başarıyla yüklendi!")

    target_column = input("Hedef değişken (target) sütununu giriniz: ")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.isnull().sum().sum() > 0:
        print("Eksik veriler tespit edildi.")
        X = X.fillna(X.median())

    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    predictions = model.predict(X)
    df['Predictions'] = predictions

    print("Tahminler eklendi ve 'Predictions' sütununa yazıldı.")

    output_file = f"predicted_{file_path.split('/')[-1]}"
    df.to_csv(output_file, index=False)
    print(f"Tahminler {output_file} dosyasına kaydedildi.")

# Ana fonksiyon
def main():
    # Dosyayı yükle
    df = load_file()
    if df is not None:
        # Modelleri oluştur ve değerlendir
        build_and_evaluate_models(df)
        print("Model değerlendirmesi tamamlandı!")

if __name__ == "__main__":
    main()
