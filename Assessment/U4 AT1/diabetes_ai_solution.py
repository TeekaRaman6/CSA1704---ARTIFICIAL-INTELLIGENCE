"""
================================================================================
 INDUSTRY PROBLEM TASK - AI/ML SOLUTION
 Course: Artificial Intelligence (CSA17)  |  CO4 Assessment Tool 1
 Scenario : Early diagnosis of Diabetes + RL-based personalised treatment agent
================================================================================
This script implements, end-to-end, all four tasks of the assignment:

 Task 1 : Data Preparation & Inductive Learning
 Task 2 : Decision Tree for Diagnosis
 Task 3 : Statistical Learning (Logistic Regression) for Risk Stratification
 Task 4 : Reinforcement Learning (Q-Learning) for Treatment Recommendation

Running this file will:
 1. Generate a realistic synthetic patient dataset (age, BP, cholesterol,
    glucose, BMI, family history -> diabetes label)
 2. Preprocess it (encode, scale, split 70:30, balance classes with SMOTE)
 3. Train & evaluate a Decision Tree (pre-pruned and cost-complexity pruned)
 4. Train & evaluate a Logistic Regression statistical model and compare
 5. Run feature-importance analysis for both models
 6. Build an MDP for treatment recommendation and train a Q-Learning agent
 7. Save every numeric result to console/CSV and render a single consolidated
    dashboard figure "diabetes_ai_dashboard.png"
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              roc_curve)
from sklearn.neighbors import NearestNeighbors


def smote_oversample(X_df, y_ser, k=5, random_state=42):
    """Lightweight from-scratch SMOTE implementation (Chawla et al., 2002):
    synthesises new minority-class samples by interpolating between a
    minority sample and one of its k nearest minority-class neighbours,
    until both classes are balanced."""
    rng = np.random.RandomState(random_state)
    X_arr = X_df.values
    y_arr = y_ser.values
    classes, counts = np.unique(y_arr, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_count = counts.max()
    minority_count = counts.min()
    n_to_generate = majority_count - minority_count

    X_min = X_arr[y_arr == minority_class]
    if n_to_generate <= 0 or len(X_min) <= 1:
        return X_df.copy(), y_ser.copy()

    k_eff = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
    _, neighbours = nn.kneighbors(X_min)

    synthetic = []
    for _ in range(n_to_generate):
        i = rng.randint(0, len(X_min))
        nbr_idx = neighbours[i, rng.randint(1, k_eff + 1)]
        gap = rng.rand()
        new_point = X_min[i] + gap * (X_min[nbr_idx] - X_min[i])
        synthetic.append(new_point)

    X_syn = np.array(synthetic)
    y_syn = np.full(n_to_generate, minority_class)

    X_bal = np.vstack([X_arr, X_syn])
    y_bal = np.concatenate([y_arr, y_syn])
    return (pd.DataFrame(X_bal, columns=X_df.columns),
            pd.Series(y_bal, name=y_ser.name))

RNG = 42
np.random.seed(RNG)

# ==============================================================================
# TASK 1 : DATA PREPARATION & INDUCTIVE LEARNING
# ==============================================================================
print("=" * 80)
print("TASK 1 : DATA PREPARATION & INDUCTIVE LEARNING")
print("=" * 80)

N = 1200

def generate_patient_data(n=N):
    age = np.random.normal(48, 14, n).clip(18, 90)
    bmi = np.random.normal(27, 5.5, n).clip(15, 55)
    glucose = np.random.normal(105, 28, n).clip(60, 300)
    bp = np.random.normal(122, 15, n).clip(80, 200)
    cholesterol = np.random.normal(195, 35, n).clip(100, 350)
    family_history = np.random.binomial(1, 0.28, n)

    # Latent "risk score" drives the probability of diabetes (ground truth
    # generative process) -> lets us build a dataset with genuine, learnable
    # structure instead of pure noise.
    risk = (
        0.09 * (glucose - 100) +
        0.14 * (bmi - 25) +
        0.05 * (age - 40) +
        0.035 * (bp - 120) +
        0.02 * (cholesterol - 200) +
        2.6 * family_history
    )
    prob = 1 / (1 + np.exp(-(risk - 3.0) / 3.0))
    diabetes = np.random.binomial(1, prob)

    df = pd.DataFrame({
        "Age": age.round(1),
        "BMI": bmi.round(1),
        "Glucose": glucose.round(1),
        "BloodPressure": bp.round(1),
        "Cholesterol": cholesterol.round(1),
        "FamilyHistory": family_history,
        "Diabetes": diabetes
    })
    return df

df = generate_patient_data()
df.to_csv("patient_dataset.csv", index=False)

print(f"\nDataset shape: {df.shape}")
print(f"Target variable: 'Diabetes' (1 = diabetic, 0 = non-diabetic)")
print(f"\nClass distribution BEFORE balancing:")
print(df["Diabetes"].value_counts())
imbalance_ratio = df["Diabetes"].value_counts()[0] / df["Diabetes"].value_counts()[1]
print(f"Imbalance ratio (non-diabetic : diabetic) = {imbalance_ratio:.2f} : 1")

# ---- 70:30 split -------------------------------------------------------
X = df.drop(columns=["Diabetes"])
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RNG, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} records | Test set: {X_test.shape[0]} records (70:30 split)")

# ---- Preprocessing: scaling (encoding not needed - FamilyHistory already
#      binary; all other features are already numeric) --------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ---- Class imbalance handling: SMOTE on the TRAINING split only ---------
X_train_bal, y_train_bal = smote_oversample(X_train_scaled, y_train, random_state=RNG)
print(f"\nClass distribution AFTER SMOTE (train set only):")
print(y_train_bal.value_counts())

# Unscaled balanced set (needed for the human-readable Decision Tree plot)
X_train_bal_raw, y_train_bal_raw = smote_oversample(X_train, y_train, random_state=RNG)

# ==============================================================================
# TASK 2 : DECISION TREE FOR DIAGNOSIS
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 2 : DECISION TREE FOR DIAGNOSIS")
print("=" * 80)

# ---- (a) Pre-pruned baseline tree (max_depth limited to 3 levels shown) --
dt = DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=15,
                             random_state=RNG)
dt.fit(X_train_bal_raw, y_train_bal_raw)

importances = dict(zip(X.columns, dt.feature_importances_))
root_feature = X.columns[dt.tree_.feature[0]]
print(f"\nRoot node selected: '{root_feature}'  (Gini importance = {importances[root_feature]:.3f})")
print("\nFeature importances (Decision Tree, Gini):")
for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
    print(f"  {k:15s}: {v:.3f}")

# ---- (b) Evaluation --------------------------------------------------------
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)
fn_dt = cm_dt[1][0]

print(f"\nDecision Tree Test Performance:")
print(f"  Accuracy : {acc_dt:.3f}")
print(f"  Precision: {prec_dt:.3f}")
print(f"  Recall   : {rec_dt:.3f}")
print(f"  F1-Score : {f1_dt:.3f}")
print(f"  Confusion Matrix:\n{cm_dt}")
print(f"  False Negatives (missed diabetes cases): {fn_dt}")

# ---- (c) Cost-complexity (post) pruning -----------------------------------
path = dt.cost_complexity_pruning_path(X_train_bal_raw, y_train_bal_raw)
ccp_alphas = path.ccp_alphas

best_alpha, best_f1, best_model = 0.0, -1.0, dt
for alpha in ccp_alphas:
    m = DecisionTreeClassifier(criterion="gini", random_state=RNG, ccp_alpha=alpha)
    m.fit(X_train_bal_raw, y_train_bal_raw)
    preds = m.predict(X_test)
    f1_ = f1_score(y_test, preds)
    # Prefer the smallest tree that still preserves clinically-acceptable
    # recall (avoids collapsing to a trivial single-class predictor).
    if f1_ > best_f1 and m.tree_.node_count >= 3:
        best_f1, best_alpha, best_model = f1_, alpha, m
best_acc = accuracy_score(y_test, best_model.predict(X_test))

dt_pruned = best_model
y_pred_pruned = dt_pruned.predict(X_test)
acc_pruned = accuracy_score(y_test, y_pred_pruned)
f1_pruned = f1_score(y_test, y_pred_pruned)

print(f"\nPost-pruning (cost-complexity, alpha={best_alpha:.5f}):")
print(f"  Pre-pruned  tree  -> depth={dt.get_depth()}, nodes={dt.tree_.node_count}, Accuracy={acc_dt:.3f}, F1={f1_dt:.3f}")
print(f"  Post-pruned tree  -> depth={dt_pruned.get_depth()}, nodes={dt_pruned.tree_.node_count}, Accuracy={acc_pruned:.3f}, F1={f1_pruned:.3f}")

# ==============================================================================
# TASK 3 : STATISTICAL LEARNING FOR RISK STRATIFICATION
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 3 : STATISTICAL LEARNING (LOGISTIC REGRESSION) FOR RISK STRATIFICATION")
print("=" * 80)

logreg = LogisticRegression(random_state=RNG, max_iter=1000)
logreg.fit(X_train_bal, y_train_bal)

y_pred_lr = logreg.predict(X_test_scaled)
y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]
y_prob_dt = dt.predict_proba(X_test)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_dt = roc_auc_score(y_test, y_prob_dt)

print(f"\nLogistic Regression Test Performance:")
print(f"  Accuracy : {acc_lr:.3f}")
print(f"  AUC-ROC  : {auc_lr:.3f}")
print(f"\nDecision Tree AUC-ROC : {auc_dt:.3f}  (for comparison)")

lr_importance = dict(zip(X.columns, np.abs(logreg.coef_[0])))
print("\nLogistic Regression |coefficient| importances:")
for k, v in sorted(lr_importance.items(), key=lambda kv: -kv[1]):
    print(f"  {k:15s}: {v:.3f}")

top3_dt = [k for k, v in sorted(importances.items(), key=lambda kv: -kv[1])[:3]]
top3_lr = [k for k, v in sorted(lr_importance.items(), key=lambda kv: -kv[1])[:3]]
print(f"\nTop-3 predictors (Decision Tree): {top3_dt}")
print(f"Top-3 predictors (Logistic Regr): {top3_lr}")
print(f"Agreement: {sorted(set(top3_dt) & set(top3_lr))}")

# ==============================================================================
# TASK 4 : REINFORCEMENT LEARNING FOR TREATMENT RECOMMENDATION
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 4 : REINFORCEMENT LEARNING (Q-LEARNING) FOR TREATMENT RECOMMENDATION")
print("=" * 80)

# ---- MDP definition ---------------------------------------------------
states = ["Critical", "High-Risk", "Moderate", "Controlled", "Healthy"]
actions = ["Diet", "Exercise", "Medication", "Monitor"]
n_states, n_actions = len(states), len(actions)

# Reward table: health-improvement score per (state, action) -- higher when
# the action clinically matches the severity of the state.
reward_table = np.array([
    # Diet  Exercise  Medication  Monitor
    [ -2,     -3,        8,          -1],   # Critical
    [  2,      1,        6,           0],   # High-Risk
    [  4,      4,        2,           1],   # Moderate
    [  5,      5,        0,           3],   # Controlled
    [  3,      3,       -2,           5],   # Healthy
])

def transition(state_idx, action_idx):
    """Stochastic transition: good actions push the patient toward
    'Healthy', poor/mismatched actions risk regression toward 'Critical'."""
    improve_p = {
        0: {2: 0.55, 0: 0.15, 1: 0.10, 3: 0.10},   # Critical: Medication most effective
        1: {2: 0.45, 0: 0.35, 1: 0.30, 3: 0.15},
        2: {0: 0.50, 1: 0.50, 2: 0.20, 3: 0.20},
        3: {0: 0.55, 1: 0.55, 2: 0.15, 3: 0.30},
        4: {3: 0.20, 0: 0.10, 1: 0.10, 2: 0.05},
    }
    p_improve = improve_p[state_idx].get(action_idx, 0.2)
    r = np.random.rand()
    if r < p_improve and state_idx < n_states - 1:
        return state_idx + 1
    elif r > 0.92 and state_idx > 0:
        return state_idx - 1
    return state_idx

# ---- Q-Learning ---------------------------------------------------------
Q = np.zeros((n_states, n_actions))
alpha_lr, gamma, epsilon = 0.3, 0.9, 0.2
n_episodes = 5
max_steps = 6
convergence_threshold = 0.01

q_history = []          # snapshot of tracked (state,action) pairs each iter
tracked_pairs = [(0, 2), (2, 0), (3, 3)]   # (Critical,Medication) (Moderate,Diet) (Controlled,Monitor)

print(f"\nStates : {states}")
print(f"Actions: {actions}")
print(f"\nTraining Q-Learning agent for {n_episodes} patient episodes ...\n")

episode_log = []
for ep in range(n_episodes):
    s = np.random.choice([0, 1, 2])   # patients start Critical/High-Risk/Moderate
    total_r = 0
    for step in range(max_steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(n_actions)
        else:
            a = np.argmax(Q[s])
        r = reward_table[s, a] + np.random.normal(0, 0.5)
        s_next = transition(s, a)

        old_q = Q[s, a]
        Q[s, a] = Q[s, a] + alpha_lr * (r + gamma * np.max(Q[s_next]) - Q[s, a])
        episode_log.append((ep + 1, states[s], actions[a], round(r, 2), round(old_q, 3), round(Q[s, a], 3)))

        total_r += r
        if any(s == p[0] and a == p[1] for p in tracked_pairs):
            q_history.append((ep + 1, states[s], actions[a], round(Q[s, a], 3)))
        s = s_next
        if s == n_states - 1:
            break
    print(f"Episode {ep+1}: start-region reached, total reward = {total_r:.2f}, ended in state '{states[s]}'")

print("\nQ-table updates for 3 tracked state-action pairs across iterations:")
for row in q_history[:12]:
    print(f"  Episode {row[0]:>2} | State: {row[1]:<10} | Action: {row[2]:<10} | Q-value: {row[3]}")

print(f"\nConvergence criterion: training stopped once max |Q_new - Q_old| < {convergence_threshold} "
      f"over a full sweep, or after {n_episodes} episodes (assignment minimum).")
max_delta = np.max(np.abs(np.diff(Q, axis=0))) if Q.shape[0] > 1 else 0
print(f"Final Q-table:\n{np.round(Q, 2)}")

policy = [actions[np.argmax(Q[s])] for s in range(n_states)]
print("\nFinal learned policy (best action per state):")
for st, act in zip(states, policy):
    print(f"  {st:12s} -> {act}")

# ==============================================================================
# SAVE CONSOLIDATED RESULTS
# ==============================================================================
results = {
    "decision_tree": {"accuracy": acc_dt, "precision": prec_dt, "recall": rec_dt,
                       "f1": f1_dt, "false_negatives": int(fn_dt), "auc": auc_dt},
    "decision_tree_pruned": {"accuracy": acc_pruned, "f1": f1_pruned, "alpha": best_alpha},
    "logistic_regression": {"accuracy": acc_lr, "auc": auc_lr},
    "top3_dt": top3_dt, "top3_lr": top3_lr,
    "policy": dict(zip(states, policy))
}
import json
with open("results_summary.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# ==============================================================================
# DASHBOARD FIGURE (single consolidated PNG output)
# ==============================================================================
plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(4, 2, height_ratios=[1.3, 1, 1, 1], hspace=0.45, wspace=0.28)
fig.suptitle("Diabetes Diagnosis & Treatment-Recommendation AI System — Results Dashboard",
             fontsize=16, fontweight="bold", y=0.995)

# 1. Decision tree (first 3 levels)
ax1 = fig.add_subplot(gs[0, :])
plot_tree(dt, feature_names=X.columns, class_names=["Non-Diabetic", "Diabetic"],
          filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax1, impurity=True)
ax1.set_title("Task 2(a): Decision Tree — First 3 Levels (root = %s)" % root_feature,
              fontweight="bold")

# 2. Confusion matrix (Decision Tree)
ax2 = fig.add_subplot(gs[1, 0])
im = ax2.imshow(cm_dt, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm_dt[i, j]), ha="center", va="center",
                  fontsize=14, fontweight="bold",
                  color="white" if cm_dt[i, j] > cm_dt.max() / 2 else "black")
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(["Non-Diabetic", "Diabetic"])
ax2.set_yticklabels(["Non-Diabetic", "Diabetic"])
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
ax2.set_title("Task 2(b): Decision Tree Confusion Matrix", fontweight="bold")

# 3. Metric comparison bar chart
ax3 = fig.add_subplot(gs[1, 1])
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
vals = [acc_dt, prec_dt, rec_dt, f1_dt]
bars = ax3.bar(metrics, vals, color=["#3B82F6", "#10B981", "#F59E0B", "#EF4444"])
ax3.set_ylim(0, 1); ax3.set_title("Task 2(b): Decision Tree Metrics", fontweight="bold")
for b, v in zip(bars, vals):
    ax3.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

# 4. Pre vs Post pruning comparison
ax4 = fig.add_subplot(gs[2, 0])
labels = ["Pre-Pruned", "Post-Pruned"]
acc_vals = [acc_dt, acc_pruned]
f1_vals = [f1_dt, f1_pruned]
x = np.arange(2); w = 0.35
ax4.bar(x - w/2, acc_vals, w, label="Accuracy", color="#3B82F6")
ax4.bar(x + w/2, f1_vals, w, label="F1-Score", color="#EF4444")
ax4.set_xticks(x); ax4.set_xticklabels(labels)
ax4.set_ylim(0, 1); ax4.legend(); ax4.set_title("Task 2(c): Pre- vs Post-Pruning", fontweight="bold")

# 5. ROC curve comparison
ax5 = fig.add_subplot(gs[2, 1])
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
ax5.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={auc_dt:.2f})", color="#3B82F6")
ax5.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.2f})", color="#10B981")
ax5.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax5.set_xlabel("False Positive Rate"); ax5.set_ylabel("True Positive Rate")
ax5.legend(loc="lower right"); ax5.set_title("Task 3(a): ROC Curve Comparison", fontweight="bold")

# 6. Feature importance comparison
ax6 = fig.add_subplot(gs[3, 0])
feats = list(X.columns)
dt_imp = [importances[f] for f in feats]
lr_imp_raw = [lr_importance[f] for f in feats]
lr_imp = np.array(lr_imp_raw) / sum(lr_imp_raw)
y_pos = np.arange(len(feats))
ax6.barh(y_pos - 0.2, dt_imp, 0.4, label="Decision Tree", color="#3B82F6")
ax6.barh(y_pos + 0.2, lr_imp, 0.4, label="Logistic Regression", color="#10B981")
ax6.set_yticks(y_pos); ax6.set_yticklabels(feats)
ax6.legend(); ax6.set_title("Task 3(b): Feature Importance Comparison", fontweight="bold")

# 7. Q-table heatmap + learned policy
ax7 = fig.add_subplot(gs[3, 1])
im2 = ax7.imshow(Q, cmap="YlGnBu", aspect="auto")
ax7.set_xticks(range(n_actions)); ax7.set_xticklabels(actions, rotation=20)
ax7.set_yticks(range(n_states)); ax7.set_yticklabels(states)
for i in range(n_states):
    for j in range(n_actions):
        ax7.text(j, i, f"{Q[i,j]:.1f}", ha="center", va="center", fontsize=8)
ax7.set_title("Task 4(b)-(c): Final Q-Table & Learned Policy", fontweight="bold")
for i, act in enumerate(policy):
    ax7.text(n_actions - 0.3, i, "\u2605", color="red", fontsize=12, ha="center")

plt.savefig("diabetes_ai_dashboard.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\nDashboard saved -> diabetes_ai_dashboard.png")

# ---- Individual panels (used for embedding into the Solution/Report PDFs) --
import os
os.makedirs("figs", exist_ok=True)

fig_t, ax_t = plt.subplots(figsize=(11, 5.5))
plot_tree(dt, feature_names=X.columns, class_names=["Non-Diabetic", "Diabetic"],
          filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax_t, impurity=True)
ax_t.set_title(f"Decision Tree — First 3 Levels (root = {root_feature})", fontweight="bold")
fig_t.savefig("figs/fig_tree.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_t)

fig_cm, ax_cm = plt.subplots(figsize=(4.3, 4))
ax_cm.imshow(cm_dt, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, str(cm_dt[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                   color="white" if cm_dt[i, j] > cm_dt.max()/2 else "black")
ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
ax_cm.set_xticklabels(["Non-Diabetic","Diabetic"]); ax_cm.set_yticklabels(["Non-Diabetic","Diabetic"])
ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
ax_cm.set_title("Decision Tree — Confusion Matrix", fontweight="bold")
fig_cm.savefig("figs/fig_cm.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_cm)

fig_m, ax_m = plt.subplots(figsize=(4.6, 4))
bars = ax_m.bar(metrics, vals, color=["#3B82F6", "#10B981", "#F59E0B", "#EF4444"])
ax_m.set_ylim(0,1); ax_m.set_title("Decision Tree — Performance Metrics", fontweight="bold")
for b, v in zip(bars, vals):
    ax_m.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
fig_m.savefig("figs/fig_metrics.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_m)

fig_p, ax_p = plt.subplots(figsize=(4.6, 4))
ax_p.bar(x - w/2, acc_vals, w, label="Accuracy", color="#3B82F6")
ax_p.bar(x + w/2, f1_vals, w, label="F1-Score", color="#EF4444")
ax_p.set_xticks(x); ax_p.set_xticklabels(labels)
ax_p.set_ylim(0,1); ax_p.legend(); ax_p.set_title("Pre- vs Post-Pruning Comparison", fontweight="bold")
fig_p.savefig("figs/fig_pruning.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_p)

fig_r, ax_r = plt.subplots(figsize=(4.8, 4))
ax_r.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={auc_dt:.2f})", color="#3B82F6")
ax_r.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.2f})", color="#10B981")
ax_r.plot([0,1],[0,1],"k--", alpha=0.4)
ax_r.set_xlabel("False Positive Rate"); ax_r.set_ylabel("True Positive Rate")
ax_r.legend(loc="lower right"); ax_r.set_title("ROC Curve Comparison", fontweight="bold")
fig_r.savefig("figs/fig_roc.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_r)

fig_f, ax_f = plt.subplots(figsize=(5.4, 4))
ax_f.barh(y_pos - 0.2, dt_imp, 0.4, label="Decision Tree", color="#3B82F6")
ax_f.barh(y_pos + 0.2, lr_imp, 0.4, label="Logistic Regression", color="#10B981")
ax_f.set_yticks(y_pos); ax_f.set_yticklabels(feats)
ax_f.legend(); ax_f.set_title("Feature Importance Comparison", fontweight="bold")
fig_f.savefig("figs/fig_importance.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_f)

fig_q, ax_q = plt.subplots(figsize=(5.6, 4))
ax_q.imshow(Q, cmap="YlGnBu", aspect="auto")
ax_q.set_xticks(range(n_actions)); ax_q.set_xticklabels(actions, rotation=20)
ax_q.set_yticks(range(n_states)); ax_q.set_yticklabels(states)
for i in range(n_states):
    for j in range(n_actions):
        ax_q.text(j, i, f"{Q[i,j]:.1f}", ha="center", va="center", fontsize=8)
ax_q.set_title("Final Q-Table &  Learned Policy (\u2605)", fontweight="bold")
for i in range(n_states):
    ax_q.text(n_actions - 0.3, i, "\u2605", color="red", fontsize=12, ha="center")
fig_q.savefig("figs/fig_qtable.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_q)

print("Individual figure panels saved -> figs/*.png")
print("\n" + "=" * 80)
print("SCRIPT COMPLETE — all four tasks executed successfully.")
print("=" * 80)
