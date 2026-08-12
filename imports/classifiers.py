import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --------- MLP Classifier ---------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze()

# ---- sklearn classifiers ----
def get_knn():
    return KNeighborsClassifier()

def get_svm():
    return SVC(probability=True)

def get_nb():
    return GaussianNB()

def get_log_reg():
    return LogisticRegression(max_iter=5000)
    
def get_rf():
    return RandomForestClassifier(random_state=42)


def train_classifier(features, labels, clf, optimizer, num_epochs=10, device='cuda'):
    loss_fn = nn.BCEWithLogitsLoss()
    val_split = 0.2

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=val_split, random_state=12
    )

    # Move data to device
    X_train = X_train.to(device).float()
    y_train = y_train.to(device).float()
    X_val = X_val.to(device).float()
    y_val = y_val.to(device).float()

    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(clf.state_dict())

    for epoch in range(num_epochs):
        clf.train()
        optimizer.zero_grad()
        logits = clf(X_train).squeeze()
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        clf.eval()
        with torch.no_grad():
            train_pred = (torch.sigmoid(logits) > 0.5).float()
            train_acc = (train_pred == y_train).float().mean().item()

            val_logits = clf(X_val).squeeze()
            val_loss = loss_fn(val_logits, y_val)
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_pred == y_val).float().mean().item()

        # Save best model if validation loss improves
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_weights = copy.deepcopy(clf.state_dict())
            print(f"New best model saved at epoch {epoch:03d} with val loss: {val_loss.item():.4f}")

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {loss.item():.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss.item():.4f}, Acc: {val_acc:.4f}"
        )

    # Load best model before returning
    clf.load_state_dict(best_model_weights)
    return clf
    

def train_and_select_classifier(opt, train_features_tensor, train_labels, device):
    """
    Unified training function for both PyTorch and sklearn classifiers.
    Returns a trained classifier (PyTorch model or sklearn estimator).
    """

    if opt.classifier == "mlp":
        clf = MLPClassifier(input_dim=train_features_tensor.shape[1]).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=opt.lr)
        clf = train_classifier(
            train_features_tensor, train_labels,
            clf, optimizer, num_epochs=11, device=device
        )
        return clf

    # ---- sklearn classifiers ----
    X_train = train_features_tensor.cpu().numpy()
    y_train = train_labels.cpu().numpy()

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    if opt.classifier == "knn":
        param_grid = {
            'clf__n_neighbors': [3, 5, 7, 9, 11],
            'clf__weights': ['uniform', 'distance']
        }
        base_clf = get_knn()

    elif opt.classifier == "svm":
        param_grid = {
            'clf__kernel': ['linear', 'rbf'],
            'clf__C': [0.1, 1, 10],
            'clf__gamma': ['scale', 'auto']
        }
        base_clf = get_svm()

    elif opt.classifier == "nb":
        param_grid = {}  # GaussianNB has no main hyperparams
        base_clf = get_nb()

    elif opt.classifier == "logreg":
        param_grid = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__solver': ['lbfgs', 'saga']
        }
        base_clf = get_log_reg()

    elif opt.classifier == "rf":
        param_grid = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_split': [2, 6, 10],
            'clf__min_samples_leaf': [1, 2]
        }
        base_clf = get_rf()

    else:
        raise ValueError(f"Unknown classifier: {opt.classifier}")

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', base_clf)
    ])

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print("Best classifier:", grid.best_estimator_)
    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    return grid.best_estimator_
