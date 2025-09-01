import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin


# Define features
categ_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# Data Preprocessing Pipeline
numeric_transf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

catego_transf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numeric_transf, num_cols),
    ("categorical", catego_transf, categ_cols)
])

# --- Custom Feature Selector ---
class RFEChi2Union(BaseEstimator, TransformerMixin):
    def __init__(self, rfe_k=10, chi2_k=10, random_state=42):
        self.rfe_k = rfe_k
        self.chi2_k = chi2_k
        self.random_state = random_state

    def fit(self, X, y):
        # Use feature indices instead of column names
        n_features = X.shape[1]
        feature_names = np.arange(n_features)

        # RFE with RandomForest
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rfe = RFE(estimator=rf, n_features_to_select=self.rfe_k, step=1)
        rfe.fit(X, y)
        self.rfe_features_ = feature_names[rfe.support_]

        # Chi^2
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        chi2_selector = SelectKBest(score_func=chi2, k=min(self.chi2_k, n_features))
        chi2_selector.fit(X_scaled, y)
        self.chi2_features_ = feature_names[chi2_selector.get_support()]

        # Final feature set (indices)
        self.selected_features_ = np.unique(np.concatenate([self.rfe_features_, self.chi2_features_]))
        return self

    def transform(self, X):
        return X[:, self.selected_features_]

# Build final pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", RFEChi2Union(rfe_k=10, chi2_k=10)),
    ("model", SVC(probability=True, random_state=42))
])