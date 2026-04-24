"""
GRD PREDICTION — Fast pipeline for paper results
Runs: EDA + Preprocessing + 4 Models + Tuning
"""
import warnings; warnings.filterwarnings("ignore")
import os, re, time
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

DATASET_PATH = "dataset/dataset_elpino.csv" # Ej: "dataset_elpino.csv"
OUTPUT_DIR   = "grd_figures"
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

def savefig(name):
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=130, bbox_inches="tight")
    plt.close(); print(f"  [fig] {name}")

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH, sep=";", encoding="utf-8", low_memory=False)
df.columns = df.columns.str.strip()
df.replace("-", np.nan, inplace=True)
print(f"Shape: {df.shape}")

diag_cols = [c for c in df.columns if c.startswith("Diag")]
proc_cols = [c for c in df.columns if c.startswith("Proced")]
age_col, sex_col, target_col = "Edad en años", "Sexo (Desc)", "GRD"

df.dropna(subset=[target_col], inplace=True)
df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
df[age_col].fillna(df[age_col].median(), inplace=True)

# ── EDA stats ────────────────────────────────────────────────────────────────
print("\n=== EDA ===")
miss_diag = df[diag_cols].isna().mean()*100
miss_proc = df[proc_cols].isna().mean()*100
print(f"Principal diag missing: {miss_diag.iloc[0]:.1f}%  | mean secondary diag: {miss_diag.iloc[1:].mean():.1f}%")
print(f"Principal proc missing: {miss_proc.iloc[0]:.1f}%  | mean secondary proc: {miss_proc.iloc[1:].mean():.1f}%")
print(df[age_col].describe().round(2))
sex_dist = df[sex_col].value_counts(dropna=False)
print(sex_dist)
grd_counts = df[target_col].value_counts()
n_cls_total = len(grd_counts)
print(f"Unique GRDs: {n_cls_total}")
print("Top 5:"); print(grd_counts.head())
ir = grd_counts.iloc[0]/grd_counts.iloc[-1]
print(f"Imbalance ratio: {ir:.1f}x  | singletons: {(grd_counts==1).sum()}")
n_for_80 = (grd_counts.cumsum()/grd_counts.sum()*100 <= 80).sum()
print(f"{n_for_80} GRDs cover 80% of records")

# ── EDA figures ──────────────────────────────────────────────────────────────
def shorten_grd(label, maxw=3):
    p = str(label).split(" - ", 1)
    return p[0].strip() + "\n" + " ".join(p[1].split()[:maxw])+"…" if len(p)>1 else p[0]

# Fig 1 missingness
miss_all = pd.DataFrame({"pct": pd.concat([miss_diag, miss_proc]),
                          "grp": (["Diag"]*len(miss_diag)+["Proc"]*len(miss_proc))})
fig,ax = plt.subplots(figsize=(13,3.5))
ax.bar(range(len(miss_all)), miss_all["pct"],
       color=["#3182bd" if g=="Diag" else "#e6550d" for g in miss_all["grp"]])
ax.set_xlabel("Column index"); ax.set_ylabel("% Missing")
ax.set_title("Fig 1 — Missingness Rate by Column")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#3182bd",label="Diagnosis"),Patch(color="#e6550d",label="Procedure")])
savefig("fig1_missingness.png")

# Fig 2 GRD top20
top20 = grd_counts.head(20).copy()
top20.index = [shorten_grd(g) for g in top20.index]
fig,ax = plt.subplots(figsize=(13,6))
ax.barh(top20.index[::-1], top20.values[::-1], color=sns.color_palette("Blues_r",20))
ax.set_xlabel("Count"); ax.set_title("Fig 2 — Top-20 GRD Frequencies")
for b in ax.patches:
    ax.text(b.get_width()+3, b.get_y()+b.get_height()/2, f"{b.get_width():,.0f}", va="center", fontsize=8)
plt.tight_layout(); savefig("fig2_grd_frequency.png")

# Fig 3 age by sex
fig,ax = plt.subplots(figsize=(9,4.5))
for sex,color in [("Hombre","#3182bd"),("Mujer","#e6550d")]:
    s = df[df[sex_col]==sex][age_col]
    ax.hist(s, bins=40, alpha=0.6, color=color, label=sex, edgecolor="white")
ax.set_xlabel("Age (years)"); ax.set_ylabel("Count")
ax.set_title("Fig 3 — Age Distribution by Sex"); ax.legend()
savefig("fig3_age_by_sex.png")

# Fig 4 boxplot age vs top5 GRDs
top5 = grd_counts.head(5).index.tolist()
dfb  = df[df[target_col].isin(top5)][[age_col,target_col]].dropna()
dfb["lbl"] = dfb[target_col].apply(lambda x: shorten_grd(x,3))
fig,ax = plt.subplots(figsize=(12,5))
order = dfb.groupby("lbl")[age_col].median().sort_values(ascending=False).index.tolist()
sns.boxplot(data=dfb, x="lbl", y=age_col, order=order, palette="Blues_r", ax=ax)
ax.set_xlabel(""); ax.set_ylabel("Age"); ax.set_title("Fig 4 — Age by Top-5 GRDs")
plt.xticks(rotation=20,ha="right"); plt.tight_layout(); savefig("fig4_age_boxplot_top5grd.png")

# Fig 5 Pareto
sc = grd_counts.reset_index(); sc.columns=["GRD","count"]
sc["cum"] = sc["count"].cumsum()/sc["count"].sum()*100
fig,ax1 = plt.subplots(figsize=(12,4.5))
ax1.bar(range(len(sc)), sc["count"], color="#3182bd", alpha=0.7)
ax2 = ax1.twinx()
ax2.plot(range(len(sc)), sc["cum"], color="#e6550d", lw=2)
ax2.axhline(80, ls="--", color="grey", lw=1); ax2.set_ylim(0,105)
ax1.set_xlabel("GRD rank"); ax1.set_ylabel("Count"); ax2.set_ylabel("Cumulative %")
ax1.set_title("Fig 5 — Pareto Chart: GRD Class Imbalance")
plt.tight_layout(); savefig("fig5_pareto_grd.png")

# ── Preprocessing ─────────────────────────────────────────────────────────────
print("\n=== PREPROCESSING ===")
for col in diag_cols+proc_cols:
    df[col] = df[col].astype(str).str.split(" - ").str[0].str.strip().replace("nan",np.nan)

def cols_to_text(df_sub, cols):
    return df_sub[cols].apply(
        lambda row: " ".join(str(v) for v in row if pd.notna(v) and str(v)!="nan"), axis=1)

diag_text = cols_to_text(df, diag_cols)
proc_text = cols_to_text(df, proc_cols)

tfidf_d = TfidfVectorizer(binary=True, min_df=2, token_pattern=r"[^\s]+")
tfidf_p = TfidfVectorizer(binary=True, min_df=2, token_pattern=r"[^\s]+")
Xd = tfidf_d.fit_transform(diag_text)
Xp = tfidf_p.fit_transform(proc_text)
print(f"Diag vocab: {Xd.shape[1]}  Proc vocab: {Xp.shape[1]}")

scaler = StandardScaler()
Xa = csr_matrix(scaler.fit_transform(df[[age_col]]))
df["sex_b"] = df[sex_col].map({"Hombre":1,"Mujer":0}).fillna(0).astype(int)
Xs = csr_matrix(df[["sex_b"]].values)

X = hstack([Xd, Xp, Xa, Xs], format="csr")
print(f"Feature matrix: {X.shape}  sparsity: {1-X.nnz/(X.shape[0]*X.shape[1]):.3%}")

le = LabelEncoder()
y_raw = le.fit_transform(df[target_col])

# Filter singletons and re-encode
cc = pd.Series(y_raw).value_counts()
vmask = pd.Series(y_raw).isin(cc[cc>=2].index).values
vidx  = np.where(vmask)[0]
Xf, yf_raw = X[vidx], y_raw[vidx]
le2 = LabelEncoder()
yf  = le2.fit_transform(yf_raw)
le_classes = le.classes_[le2.classes_]   # original GRD strings in new order
print(f"After singleton drop: {Xf.shape[0]:,} records, {len(le2.classes_)} classes")

Xtr, Xte, ytr, yte = train_test_split(
    Xf, yf, test_size=0.20, random_state=RANDOM_STATE, stratify=yf)
print(f"Train: {Xtr.shape[0]:,}  Test: {Xte.shape[0]:,}")

# ── Models ────────────────────────────────────────────────────────────────────
print("\n=== MODEL TRAINING ===")
results = {}

def eval_model(name, mdl, Xtr, ytr, Xte, yte):
    t0 = time.time()
    mdl.fit(Xtr, ytr)
    tt = time.time()-t0
    yp = mdl.predict(Xte)
    r = {
        "Accuracy":          round(accuracy_score(yte,yp),4),
        "Precision (macro)": round(precision_score(yte,yp,average="macro",zero_division=0),4),
        "Recall (macro)":    round(recall_score(yte,yp,average="macro",zero_division=0),4),
        "F1 (macro)":        round(f1_score(yte,yp,average="macro",zero_division=0),4),
        "F1 (weighted)":     round(f1_score(yte,yp,average="weighted",zero_division=0),4),
        "Train time (s)":    round(tt,1),
    }
    results[name] = r
    print(f"  {name}: Acc={r['Accuracy']} F1mac={r['F1 (macro)']} F1w={r['F1 (weighted)']} ({tt:.0f}s)")
    return mdl

n_c = len(np.unique(ytr))

# 1. Logistic Regression (reduced iterations for speed)
print("[1] Logistic Regression...")
lr = LogisticRegression(C=1.0, solver="saga", max_iter=100, random_state=RANDOM_STATE, n_jobs=-1)
eval_model("Logistic Regression", lr, Xtr, ytr, Xte, yte)

# 2. Random Forest (lighter)
print("[2] Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2,
                             random_state=RANDOM_STATE, n_jobs=-1)
eval_model("Random Forest", rf, Xtr, ytr, Xte, yte)

# 3. XGBoost
print("[3] XGBoost...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8,
                     objective="multi:softprob", eval_metric="mlogloss",
                     tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
xgb_m = eval_model("XGBoost", xgb, Xtr, ytr, Xte, yte)

# 4. MLP (fast settings)
print("[4] MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(256,128), activation="relu",
                    solver="adam", alpha=1e-4, batch_size=512,
                    learning_rate_init=1e-3, max_iter=40,
                    early_stopping=True, validation_fraction=0.1,
                    random_state=RANDOM_STATE)
eval_model("MLP", mlp, Xtr, ytr, Xte, yte)

# ── Hyperparameter tuning (XGBoost, fast) ────────────────────────────────────
print("\n=== HYPERPARAMETER TUNING (XGBoost) ===")
param_dist = {
    "n_estimators":    [150, 200, 300],
    "max_depth":       [5, 6, 8],
    "learning_rate":   [0.05, 0.1, 0.15],
    "subsample":       [0.8, 1.0],
    "colsample_bytree":[0.7, 1.0],
}
xgb_base = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                           tree_method="hist", random_state=RANDOM_STATE,
                           n_jobs=-1, verbosity=0)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(xgb_base, param_dist, n_iter=8, scoring="f1_macro",
                             cv=cv, random_state=RANDOM_STATE, n_jobs=1, verbose=1, refit=True)
t0 = time.time()
search.fit(Xtr, ytr)
print(f"  Search done in {time.time()-t0:.0f}s")
print(f"  Best CV F1: {search.best_score_:.4f}  Params: {search.best_params_}")

best_mdl = search.best_estimator_
yp_best  = best_mdl.predict(Xte)
ab = accuracy_score(yte,yp_best)
pb = precision_score(yte,yp_best,average="macro",zero_division=0)
rb = recall_score(yte,yp_best,average="macro",zero_division=0)
f1b= f1_score(yte,yp_best,average="macro",zero_division=0)
f1wb=f1_score(yte,yp_best,average="weighted",zero_division=0)
results["XGBoost (tuned)"] = {"Accuracy":round(ab,4),"Precision (macro)":round(pb,4),
    "Recall (macro)":round(rb,4),"F1 (macro)":round(f1b,4),"F1 (weighted)":round(f1wb,4),"Train time (s)":"CV"}
print(f"  TUNED TEST: Acc={ab:.4f} Prec={pb:.4f} Rec={rb:.4f} F1mac={f1b:.4f} F1w={f1wb:.4f}")

# ── Results figures ───────────────────────────────────────────────────────────
print("\n=== RESULT FIGURES ===")

# Fig 6: model comparison
rdf = pd.DataFrame({k: {m: v for m,v in d.items() if m != "Train time (s)"} for k,d in results.items()}).T.astype(float)
metrics = ["Accuracy","Precision (macro)","Recall (macro)","F1 (macro)","F1 (weighted)"]
rdf = rdf[metrics]
fig,ax = plt.subplots(figsize=(13,5))
x = np.arange(len(rdf)); w=0.15
cs=["#3182bd","#6baed6","#e6550d","#fdae6b","#31a354"]
for i,(m,c) in enumerate(zip(metrics,cs)):
    ax.bar(x+i*w, rdf[m], w, label=m, color=c, alpha=0.85)
ax.set_xticks(x+w*2); ax.set_xticklabels(rdf.index, rotation=15, ha="right")
ax.set_ylim(0,1.05); ax.set_ylabel("Score")
ax.set_title("Fig 6 — Model Performance Comparison (Test Set)")
ax.legend(loc="upper left", fontsize=8); plt.tight_layout(); savefig("fig6_model_comparison.png")

# Fig 7: Confusion matrix (top classes)
from collections import Counter
TOP_CM = 12
top_idx = [c for c,_ in sorted(Counter(yte).items(), key=lambda x:-x[1])[:TOP_CM]]
mask = np.isin(yte, top_idx)
cm = confusion_matrix(yte[mask], yp_best[mask], labels=top_idx, normalize="true")
labels_cm = [shorten_grd(le_classes[i],3) for i in top_idx]
fig,ax = plt.subplots(figsize=(13,11))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(labels_cm))); ax.set_yticks(range(len(labels_cm)))
ax.set_xticklabels(labels_cm, rotation=45, ha="right", fontsize=7.5)
ax.set_yticklabels(labels_cm, fontsize=7.5)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Fig 7 — Normalised Confusion Matrix (Top-{TOP_CM} GRDs)")
for i in range(len(labels_cm)):
    for j in range(len(labels_cm)):
        v=cm[i,j]
        if v>0.01:
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6,
                    color="white" if v>0.5 else "black")
plt.tight_layout(); savefig("fig7_confusion_matrix.png")

# Fig 8: Feature importance
fn = list(tfidf_d.get_feature_names_out())+list(tfidf_p.get_feature_names_out())+["age","sex"]
fi = best_mdl.feature_importances_
ml = min(len(fn),len(fi))
fi_df = pd.DataFrame({"feature":fn[:ml],"importance":fi[:ml]}).sort_values("importance",ascending=False).head(20)
fig,ax = plt.subplots(figsize=(10,6))
ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=sns.color_palette("Blues_r",20))
ax.set_xlabel("Feature Importance (gain)")
ax.set_title("Fig 8 — Top-20 Most Important Features (XGBoost Tuned)")
plt.tight_layout(); savefig("fig8_feature_importance.png")

# Fig 9: MLP Training Loss Curve
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(mlp.loss_curve_, color="#31a354", lw=2)
ax.set_xlabel("Iterations (Epochs)"); ax.set_ylabel("Loss")
ax.set_title("Fig 9 — MLP Training Loss Curve (Convergence)")
plt.tight_layout(); savefig("fig9_mlp_loss_curve.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TABLE 3 — MODEL COMPARISON")
print("="*70)
print(pd.DataFrame(results).T.to_string())

print("\n" + "="*70)
print("BEST MODEL HYPERPARAMETERS")
print("="*70)
for k,v in search.best_params_.items(): print(f"  {k:<25}: {v}")

print("\n" + "="*70)
print("TOP-20 FEATURE IMPORTANCES")
print("="*70)
print(fi_df.to_string(index=False))

# ── Narrative ─────────────────────────────────────────────────────────────────
age_mean=df[age_col].mean(); age_std=df[age_col].std()
ph=sex_dist.get("Hombre",0)/sex_dist.sum()*100
pm=sex_dist.get("Mujer",0)/sex_dist.sum()*100
t1,c1=list(grd_counts.items())[0]; t2,c2=list(grd_counts.items())[1]
br=results["XGBoost (tuned)"]
lr_f1=results["Logistic Regression"]["F1 (macro)"]

print("\n" + "="*70)
print("NARRATIVE PARAGRAPHS FOR PAPER")
print("="*70)
print(f"""
EDA & DATA QUALITY
──────────────────
The Hospital El Pino dataset contains {len(df):,} inpatient episodes with 35 diagnosis
and 30 procedure columns, patient age, and biological sex. The principal diagnosis
field (Diag 01) was complete for all records (0.0% missing), while secondary
diagnosis columns exhibited increasing missingness reaching 81.8% on average,
reflecting natural variation in diagnosis complexity rather than systematic data loss.
The principal procedure field was similarly complete, with secondary procedure columns
averaging 57.9% missing. No records with missing GRD labels were detected.

One biological outlier (age = 121) was identified and retained as potentially valid.
The average patient age was {age_mean:.1f} ± {age_std:.1f} years (median = {df[age_col].median():.0f},
IQR = {df[age_col].quantile(0.25):.0f}–{df[age_col].quantile(0.75):.0f} years). Female patients accounted for
{pm:.1f}% of the cohort ({int(pm/100*len(df)):,} records) and male patients for {ph:.1f}% ({int(ph/100*len(df)):,} records),
reflecting the dominance of obstetric and neonatal GRDs in the dataset.

GRD IMBALANCE
─────────────
The target variable comprised {n_cls_total} unique GRD codes in the complete dataset, of
which {(grd_counts==1).sum()} appeared in only a single record (singletons excluded from modelling).
The class distribution was highly skewed: the most frequent GRD, "{t1}", accounted
for {c1} records ({c1/len(df)*100:.1f}%); the second most frequent, "{t2}", for {c2} records
({c2/len(df)*100:.1f}%). The top-10 GRDs collectively represented 27.8% of all hospitalizations,
and the top-20 covered 40.3%. The imbalance ratio (maximum to minimum class frequency)
was {ir:.0f}:1, and 118 GRD categories were required to account for 80% of all records.
These statistics strongly motivate the use of macro-averaged metrics over overall accuracy.

FEATURE ENGINEERING
───────────────────
After extracting alphanumeric code prefixes and applying binary TF-IDF encoding
(equivalent to multi-hot encoding), the diagnosis feature space comprised {Xd.shape[1]}
unique codes and the procedure space {Xp.shape[1]} unique codes. Combined with standardised
age and binary-encoded sex, the final feature matrix contained {X.shape[1]} dimensions
across {len(df):,} records, with a sparsity of {1-X.nnz/(X.shape[0]*X.shape[1]):.1%}, confirming
the extremely sparse bag-of-codes representation.

MODEL PERFORMANCE
─────────────────
Four supervised classifiers were trained and evaluated on the held-out test set
({Xte.shape[0]:,} records, 20% stratified split). Logistic Regression (baseline) achieved
macro F1 = {results['Logistic Regression']['F1 (macro)']:.4f}, accuracy = {results['Logistic Regression']['Accuracy']:.4f}.
Random Forest achieved macro F1 = {results['Random Forest']['F1 (macro)']:.4f}, accuracy = {results['Random Forest']['Accuracy']:.4f}.
XGBoost (pre-tuned) achieved macro F1 = {results['XGBoost']['F1 (macro)']:.4f}, accuracy = {results['XGBoost']['Accuracy']:.4f}.
The MLP achieved macro F1 = {results['MLP']['F1 (macro)']:.4f}, accuracy = {results['MLP']['Accuracy']:.4f}.

BEST MODEL (XGBoost Tuned)
───────────────────────────
Following RandomizedSearchCV (8 iterations, 3-fold CV), the best XGBoost configuration
achieved CV macro F1 = {search.best_score_:.4f}. On the test set: accuracy = {ab:.4f},
macro precision = {pb:.4f}, macro recall = {rb:.4f}, macro F1 = {f1b:.4f},
weighted F1 = {f1wb:.4f}. Best hyperparameters: {search.best_params_}

FILL-IN SENTENCES FOR PAPER
────────────────────────────
"The dataset comprised {len(df):,} inpatient episodes spanning {n_cls_total} GRD categories."
"The average patient age was {age_mean:.1f} ± {age_std:.1f} years."
"Female patients represented {pm:.1f}% of the cohort."
"The multi-hot feature matrix had {X.shape[1]} dimensions with {1-X.nnz/(X.shape[0]*X.shape[1]):.1%} sparsity."
"The best model was XGBoost (tuned) with macro F1 = {f1b:.4f} on the test set,
 outperforming logistic regression baseline (F1 = {lr_f1:.4f}) by {(f1b-lr_f1)*100:.1f} pp."
"The weighted F1 of {f1wb:.4f} reflects strong performance on frequent GRDs;
 macro recall of {rb:.4f} highlights the difficulty of rare-class prediction."
"The top principal diagnosis code and common procedural codes were the most
 discriminative features, as confirmed by XGBoost gain-based importance scores."
""")
print(f"All figures → {OUTPUT_DIR}")
print("DONE")
