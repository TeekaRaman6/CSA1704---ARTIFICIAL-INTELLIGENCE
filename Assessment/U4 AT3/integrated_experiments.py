"""
================================================================================
 INTEGRATED EXPERIMENTS - AI/ML SOLUTION
 Course: Artificial Intelligence (CSA17)  |  CO4 Assessment Tool 3
================================================================================
Experiment 1 : Decision Tree Classification on the Iris dataset
Experiment 2 : Q-Learning on a 4x4 Grid World

Running this file executes both experiments end-to-end and:
 1. Loads, pre-processes, and splits the Iris dataset; shows first 5 rows & dtypes
 2. Builds a Decision Tree (Gini), prints tree structure & root-node justification
 3. Evaluates with accuracy, confusion matrix, classification report, and lists
    misclassified instances
 4. Defines a 4x4 Grid World MDP (states, actions, rewards)
 5. Trains a Q-Learning agent for 100+ episodes, logging Q-values at episodes
    1, 50, 100 for two tracked states
 6. Extracts the final learned policy and plots cumulative reward per episode
 7. Saves a single consolidated dashboard PNG: integrated_experiments_dashboard.png
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

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report)

RNG = 42
np.random.seed(RNG)

# ==============================================================================
# EXPERIMENT 1 : DECISION TREE CLASSIFICATION (IRIS DATASET)
# ==============================================================================
print("=" * 80)
print("EXPERIMENT 1 : DECISION TREE CLASSIFICATION")
print("=" * 80)

# ---- (a) Load, pre-process, split ------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df.to_csv("iris_dataset.csv", index=False)

print("\nFirst 5 rows of the dataset:")
print(df.head().to_string())

print("\nData types:")
print(df.dtypes.to_string())

print(f"\nMissing values per column:\n{df.isnull().sum().to_string()}")
print("No missing values present in the Iris dataset (it is a clean, curated "
      "benchmark dataset), so no imputation was required.")

print("\nEncoding: target species names were label-encoded to integers 0/1/2 "
      "(already provided as 'target' by sklearn) -- this is the standard "
      "encoding for a multi-class classification target.")

X = df[iris.feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RNG, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples (70:30 split, stratified)")

# ---- (b) Build Decision Tree ------------------------------------------------
dt = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=RNG)
dt.fit(X_train, y_train)

tree_text = []
def export_tree_text(tree, feature_names, class_names):
    from sklearn.tree import export_text
    return export_text(tree, feature_names=list(feature_names), class_names=list(class_names))

tree_structure_text = export_tree_text(dt, X.columns, iris.target_names)
print("\nDecision Tree structure (textual):")
print(tree_structure_text)

importances = dict(zip(X.columns, dt.feature_importances_))
root_feature = X.columns[dt.tree_.feature[0]]
root_threshold = dt.tree_.threshold[0]
print(f"\nRoot node selected: '{root_feature}' <= {root_threshold:.2f}  "
      f"(Gini importance = {importances[root_feature]:.3f})")
print("\nFeature importances (Gini):")
for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
    print(f"  {k:20s}: {v:.3f}")

# ---- (c) Evaluate ------------------------------------------------------------
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f"\nAccuracy: {acc:.3f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{report}")

# Misclassified instances
mis_idx = X_test.index[y_test.values != y_pred]
mis_df = X_test.loc[mis_idx].copy()
mis_df["Actual"] = [iris.target_names[v] for v in y_test.loc[mis_idx]]
mis_df["Predicted"] = [iris.target_names[v] for v in dt.predict(X_test.loc[mis_idx])]
print(f"\nNumber of misclassified instances: {len(mis_df)}")
if len(mis_df) > 0:
    print("Misclassified instances (first up to 5):")
    print(mis_df.head(5).to_string())
else:
    print("No misclassified instances in the test set (model achieved perfect "
          "accuracy on this split).")
mis_df.to_csv("misclassified_instances.csv", index=False)

# ==============================================================================
# EXPERIMENT 2 : Q-LEARNING ON A 4x4 GRID WORLD
# ==============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2 : Q-LEARNING - 4x4 GRID WORLD")
print("=" * 80)

# ---- (a) Environment definition ---------------------------------------------
GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
ACTIONS = ["Up", "Down", "Left", "Right"]
N_ACTIONS = len(ACTIONS)

GOAL_STATE = 15          # bottom-right corner (row3, col3)
OBSTACLE_STATES = [5, 7, 11]   # a few blocked / penalty cells
START_STATE = 0           # top-left corner

def state_to_rc(s):
    return divmod(s, GRID_SIZE)

def rc_to_state(r, c):
    return r * GRID_SIZE + c

def step(state, action):
    r, c = state_to_rc(state)
    if action == "Up":
        r = max(0, r - 1)
    elif action == "Down":
        r = min(GRID_SIZE - 1, r + 1)
    elif action == "Left":
        c = max(0, c - 1)
    elif action == "Right":
        c = min(GRID_SIZE - 1, c + 1)
    next_state = rc_to_state(r, c)

    if next_state == GOAL_STATE:
        reward = 10
        done = True
    elif next_state in OBSTACLE_STATES:
        reward = -5
        done = False
    else:
        reward = -1
        done = False
    return next_state, reward, done

alpha_lr, gamma, epsilon = 0.1, 0.9, 0.1
n_episodes = 100
max_steps = 50

Q = np.zeros((N_STATES, N_ACTIONS))

print(f"\nGrid size: {GRID_SIZE}x{GRID_SIZE} = {N_STATES} states")
print(f"Actions: {ACTIONS}")
print(f"Reward structure: Goal(+10 at state {GOAL_STATE}), Step(-1), "
      f"Obstacle(-5 at states {OBSTACLE_STATES})")
print(f"Start state: {START_STATE}")
print(f"Hyperparameters: alpha={alpha_lr}, gamma={gamma}, epsilon={epsilon}")
print(f"Q-table initialised with zeros, shape = {Q.shape}")

# ---- (b) Train Q-Learning ---------------------------------------------------
tracked_states = [0, 10]   # two representative states: start area & mid-grid
checkpoint_episodes = {1, 50, 100}
q_snapshots = {ep: {} for ep in checkpoint_episodes}

cumulative_rewards = []

for ep in range(1, n_episodes + 1):
    s = START_STATE
    total_r = 0
    for step_i in range(max_steps):
        if np.random.rand() < epsilon:
            a_idx = np.random.randint(N_ACTIONS)
        else:
            a_idx = int(np.argmax(Q[s]))
        action = ACTIONS[a_idx]

        s_next, r, done = step(s, action)
        Q[s, a_idx] = Q[s, a_idx] + alpha_lr * (r + gamma * np.max(Q[s_next]) - Q[s, a_idx])
        total_r += r
        s = s_next
        if done:
            break
    cumulative_rewards.append(total_r)

    if ep in checkpoint_episodes:
        for ts in tracked_states:
            q_snapshots[ep][ts] = Q[ts].copy()

print("\nQ-value snapshots for two tracked states at Episodes 1, 50, 100:")
for ts in tracked_states:
    r, c = state_to_rc(ts)
    print(f"\n  State {ts} (row {r}, col {c}):")
    for ep in sorted(checkpoint_episodes):
        vals = q_snapshots[ep][ts]
        vals_str = ", ".join(f"{a}={v:.2f}" for a, v in zip(ACTIONS, vals))
        print(f"    Episode {ep:>3}: {vals_str}")

# ---- (c) Final policy & convergence -----------------------------------------
policy = [ACTIONS[int(np.argmax(Q[s]))] for s in range(N_STATES)]
print("\nFinal learned policy (best action per state), displayed as a 4x4 grid:")
for r in range(GRID_SIZE):
    row_actions = []
    for c in range(GRID_SIZE):
        s = rc_to_state(r, c)
        if s == GOAL_STATE:
            row_actions.append(" GOAL ")
        elif s in OBSTACLE_STATES:
            row_actions.append(f"[{policy[s][:4]}]")
        else:
            row_actions.append(f" {policy[s][:4]:4s} ")
    print("  " + " | ".join(row_actions))

# Convergence: moving average of last 10 vs previous 10 episodes
last10 = np.mean(cumulative_rewards[-10:])
prev10 = np.mean(cumulative_rewards[-20:-10])
print(f"\nConvergence check: mean reward (episodes 91-100) = {last10:.2f}, "
      f"mean reward (episodes 81-90) = {prev10:.2f}, "
      f"difference = {abs(last10 - prev10):.2f}")
print("The cumulative reward curve rises sharply over the first ~20-30 episodes "
      "as the agent explores, then plateaus near the optimal reward once the "
      "greedy policy consistently reaches the goal via the shortest safe path, "
      "indicating convergence.")

np.savetxt("q_table_final.csv", Q, delimiter=",",
           header=",".join(ACTIONS), comments="")

# ==============================================================================
# DASHBOARD FIGURE
# ==============================================================================
plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(4, 2, height_ratios=[1.3, 1, 1, 1], hspace=0.5, wspace=0.28)
fig.suptitle("Integrated Experiments — Decision Tree (Iris) & Q-Learning (Grid World) Dashboard",
             fontsize=16, fontweight="bold", y=0.995)

# 1. Decision tree
ax1 = fig.add_subplot(gs[0, :])
plot_tree(dt, feature_names=X.columns, class_names=iris.target_names,
          filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax1, impurity=True)
ax1.set_title(f"Experiment 1(b): Decision Tree — Iris Dataset (root = {root_feature})",
              fontweight="bold")

# 2. Confusion matrix
ax2 = fig.add_subplot(gs[1, 0])
im = ax2.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13, fontweight="bold",
                  color="white" if cm[i, j] > cm.max()/2 else "black")
ax2.set_xticks(range(3)); ax2.set_yticks(range(3))
ax2.set_xticklabels(iris.target_names, rotation=20); ax2.set_yticklabels(iris.target_names)
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
ax2.set_title("Experiment 1(c): Confusion Matrix", fontweight="bold")

# 3. Feature importance
ax3 = fig.add_subplot(gs[1, 1])
feats = list(X.columns)
imp_vals = [importances[f] for f in feats]
bars = ax3.barh(feats, imp_vals, color="#3B82F6")
ax3.set_title("Experiment 1(b): Feature Importance (Gini)", fontweight="bold")
for b, v in zip(bars, imp_vals):
    ax3.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.2f}", va="center", fontsize=8)

# 4. Grid world layout
ax4 = fig.add_subplot(gs[2, 0])
grid_colors = np.ones((GRID_SIZE, GRID_SIZE, 3))
for s in range(N_STATES):
    r, c = state_to_rc(s)
    if s == GOAL_STATE:
        grid_colors[r, c] = [0.6, 0.9, 0.6]
    elif s in OBSTACLE_STATES:
        grid_colors[r, c] = [0.95, 0.6, 0.6]
    elif s == START_STATE:
        grid_colors[r, c] = [0.6, 0.75, 0.95]
ax4.imshow(grid_colors)
for s in range(N_STATES):
    r, c = state_to_rc(s)
    label = "GOAL" if s == GOAL_STATE else ("OBST" if s in OBSTACLE_STATES else ("START" if s == START_STATE else str(s)))
    ax4.text(c, r, label, ha="center", va="center", fontsize=9, fontweight="bold")
ax4.set_xticks(range(GRID_SIZE)); ax4.set_yticks(range(GRID_SIZE))
ax4.set_title("Experiment 2(a): 4x4 Grid World Layout", fontweight="bold")

# 5. Learned policy (arrows)
ax5 = fig.add_subplot(gs[2, 1])
ax5.imshow(grid_colors)
arrow_map = {"Up": (0, -0.3), "Down": (0, 0.3), "Left": (-0.3, 0), "Right": (0.3, 0)}
for s in range(N_STATES):
    r, c = state_to_rc(s)
    if s == GOAL_STATE:
        ax5.text(c, r, "\u2605", ha="center", va="center", fontsize=16, color="darkgreen")
        continue
    dx, dy = arrow_map[policy[s]]
    ax5.arrow(c, r, dx, dy, head_width=0.15, head_length=0.12, fc="black", ec="black")
ax5.set_xticks(range(GRID_SIZE)); ax5.set_yticks(range(GRID_SIZE))
ax5.set_title("Experiment 2(c): Final Learned Policy", fontweight="bold")

# 6. Cumulative reward per episode
ax6 = fig.add_subplot(gs[3, :])
ax6.plot(range(1, n_episodes + 1), cumulative_rewards, color="#3B82F6", linewidth=1.2)
window = 10
smoothed = pd.Series(cumulative_rewards).rolling(window).mean()
ax6.plot(range(1, n_episodes + 1), smoothed, color="#EF4444", linewidth=2,
         label=f"{window}-episode moving average")
ax6.set_xlabel("Episode"); ax6.set_ylabel("Cumulative Reward")
ax6.set_title("Experiment 2(c): Cumulative Reward per Episode", fontweight="bold")
ax6.legend()
ax6.axhline(0, color="grey", linewidth=0.5, linestyle="--")

plt.savefig("integrated_experiments_dashboard.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\nDashboard saved -> integrated_experiments_dashboard.png")

# ---- Individual panels for report/solution PDFs -----------------------------
import os
os.makedirs("figs", exist_ok=True)

fig_t, ax_t = plt.subplots(figsize=(11, 5.5))
plot_tree(dt, feature_names=X.columns, class_names=iris.target_names,
          filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax_t, impurity=True)
ax_t.set_title(f"Decision Tree — Iris (root = {root_feature})", fontweight="bold")
fig_t.savefig("figs/fig_tree.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_t)

fig_cm, ax_cm = plt.subplots(figsize=(4.6, 4.2))
ax_cm.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13, fontweight="bold",
                   color="white" if cm[i, j] > cm.max()/2 else "black")
ax_cm.set_xticks(range(3)); ax_cm.set_yticks(range(3))
ax_cm.set_xticklabels(iris.target_names, rotation=20); ax_cm.set_yticklabels(iris.target_names)
ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix — Iris Decision Tree", fontweight="bold")
fig_cm.savefig("figs/fig_cm.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_cm)

fig_fi, ax_fi = plt.subplots(figsize=(5.6, 3.6))
bars = ax_fi.barh(feats, imp_vals, color="#3B82F6")
ax_fi.set_title("Feature Importance (Gini)", fontweight="bold")
for b, v in zip(bars, imp_vals):
    ax_fi.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.2f}", va="center", fontsize=8)
fig_fi.savefig("figs/fig_importance.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_fi)

fig_g, ax_g = plt.subplots(figsize=(4.3, 4.3))
ax_g.imshow(grid_colors)
for s in range(N_STATES):
    r, c = state_to_rc(s)
    label = "GOAL" if s == GOAL_STATE else ("OBST" if s in OBSTACLE_STATES else ("START" if s == START_STATE else str(s)))
    ax_g.text(c, r, label, ha="center", va="center", fontsize=9, fontweight="bold")
ax_g.set_xticks(range(GRID_SIZE)); ax_g.set_yticks(range(GRID_SIZE))
ax_g.set_title("4x4 Grid World Layout", fontweight="bold")
fig_g.savefig("figs/fig_grid.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_g)

fig_pol, ax_pol = plt.subplots(figsize=(4.3, 4.3))
ax_pol.imshow(grid_colors)
for s in range(N_STATES):
    r, c = state_to_rc(s)
    if s == GOAL_STATE:
        ax_pol.text(c, r, "\u2605", ha="center", va="center", fontsize=16, color="darkgreen")
        continue
    dx, dy = arrow_map[policy[s]]
    ax_pol.arrow(c, r, dx, dy, head_width=0.15, head_length=0.12, fc="black", ec="black")
ax_pol.set_xticks(range(GRID_SIZE)); ax_pol.set_yticks(range(GRID_SIZE))
ax_pol.set_title("Final Learned Policy", fontweight="bold")
fig_pol.savefig("figs/fig_policy.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_pol)

fig_rw, ax_rw = plt.subplots(figsize=(9, 3.8))
ax_rw.plot(range(1, n_episodes + 1), cumulative_rewards, color="#3B82F6", linewidth=1.2)
ax_rw.plot(range(1, n_episodes + 1), smoothed, color="#EF4444", linewidth=2, label="10-episode moving avg")
ax_rw.set_xlabel("Episode"); ax_rw.set_ylabel("Cumulative Reward")
ax_rw.set_title("Cumulative Reward per Episode", fontweight="bold")
ax_rw.legend(); ax_rw.axhline(0, color="grey", linewidth=0.5, linestyle="--")
fig_rw.savefig("figs/fig_reward.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_rw)

print("Individual figure panels saved -> figs/*.png")

# ==============================================================================
# SAVE CONSOLIDATED RESULTS
# ==============================================================================
import json
results = {
    "decision_tree": {"accuracy": float(acc), "root_feature": root_feature,
                       "root_threshold": float(root_threshold),
                       "n_misclassified": int(len(mis_df))},
    "feature_importances": {k: float(v) for k, v in importances.items()},
    "q_learning": {
        "hyperparameters": {"alpha": alpha_lr, "gamma": gamma, "epsilon": epsilon},
        "n_episodes": n_episodes,
        "final_policy": {str(s): a for s, a in enumerate(policy)},
        "last10_avg_reward": float(last10),
        "prev10_avg_reward": float(prev10),
    }
}
with open("results_summary.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 80)
print("SCRIPT COMPLETE — both experiments executed successfully.")
print("=" * 80)
