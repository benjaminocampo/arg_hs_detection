# %% [markdown]
# Argument-base hate speech detection using premises and conclusion &
# check-worthiness labels. Preprocessing is done with one-hot encoding.
# %%
import pandas as pd

df = pd.read_csv("../data/wsf_arg_plus_per_message.csv")
# %%
df
# %%
columns = [
    "premise0",
    "premise1",
    "premise2",
    "premise3",
    "premise4",
    "premise5",
    "conclusion"
]
# %%
cw_cols = [f"{c}_cw_platinum" for c in columns]
hate_cols = [f"{c}_hate" for c in columns]
df_st = df[["is_argument", "concat_hate"] + cw_cols + hate_cols]
# %%
df_arg = df_st[df_st["is_argument"] == "yes"].copy()
# %%
df_arg
# %%
for c in cw_cols:
    df_arg[c] = pd.Categorical(df_arg[c], categories=["NFS", "UFS", "CFS"])
#df_arg[cw_cols] = pd.Categorical(df[cw_cols], categories=["NFS", "UFS", "CFS"])
# %%
df_arg_features = pd.get_dummies(df_arg[cw_cols]).astype(int)
# %%
df_arg_features = pd.concat([df_arg[cw_cols].notna().astype(int), df_arg[hate_cols].fillna(0).astype(int), df_arg_features], axis=1)
# %%
df_arg_features
# %%
# Set order
df_arg_features = df_arg_features[[
   "premise0_cw_platinum",
   "premise0_hate", 
   "premise0_cw_platinum_CFS",
   "premise0_cw_platinum_NFS",
   "premise0_cw_platinum_UFS",
   "premise1_cw_platinum",
   "premise1_hate",
   "premise1_cw_platinum_CFS",
   "premise1_cw_platinum_NFS",
   "premise1_cw_platinum_UFS",
   "premise2_cw_platinum",
   "premise2_hate",
   "premise2_cw_platinum_CFS",
   "premise2_cw_platinum_NFS",
   "premise2_cw_platinum_UFS",
   "premise3_cw_platinum",
   "premise3_hate",
   "premise3_cw_platinum_CFS",
   "premise3_cw_platinum_NFS",
   "premise3_cw_platinum_UFS",
   "premise4_cw_platinum",
   "premise4_hate",
   "premise4_cw_platinum_CFS",
   "premise4_cw_platinum_NFS",
   "premise4_cw_platinum_UFS",
   "premise5_cw_platinum",
   "premise5_hate",
   "premise5_cw_platinum_CFS",
   "premise5_cw_platinum_NFS",
   "premise5_cw_platinum_UFS",
   "conclusion_cw_platinum",
   "conclusion_hate",
   "conclusion_cw_platinum_CFS",
   "conclusion_cw_platinum_NFS",
   "conclusion_cw_platinum_UFS",
]]
# %%
df_arg_features
# %% [markdown]
# Experiment 1: Test on cross-validation using 5 folds.
# Using everything Structure + Check-worthiness labels.
# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

X = df_arg_features.drop(columns=hate_cols)
y = df_arg["concat_hate"]

def kfoldres(X, y):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    models = {
        "lgr": LogisticRegression(max_iter=1000),
        "rforest": RandomForestClassifier(),
        "svm": SVC(),
        "xgb": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=0
        ),
    }

    results = {}

    for split_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            p, r, f1, sup = precision_recall_fscore_support(y_test, preds, average="macro")
            results[f"kf={split_idx}_{model_name}"] = {}
            results[f"kf={split_idx}_{model_name}"]["p"] = p
            results[f"kf={split_idx}_{model_name}"]["r"] = r
            results[f"kf={split_idx}_{model_name}"]["f1"] = f1
            #results[f"kf={split_idx}_{model_name}"]["sup"] = sup
    
    return results
# %%
results = kfoldres(X, y)
df_results = pd.DataFrame(results).T
# %%
df_results = df_results.reset_index().rename(columns={"index": "kfold"})
# %%
df_results["model_name"] = df_results["kfold"].apply(lambda fold: fold.split("_")[1])
df_results["kfold"] = df_results["kfold"].apply(lambda fold: fold.split("_")[0])
# %%
df_arg_str_cw = (
    df_results
    .drop(columns=["kfold"])
    .groupby("model_name")
    .agg(["mean", "std"])
    .round(3)
)
# %%
df_arg_str_cw
# %%
# %% [markdown]
# Experiment 2: Test on cross-validation using 5 folds.
# Using Structure only
# %%
X_st = df_arg_features[cw_cols]
# %%
results_st = kfoldres(X_st, y)
df_results_st = pd.DataFrame(results_st).T
# %%
df_results_st = df_results_st.reset_index().rename(columns={"index": "kfold"})
# %%
df_results_st["model_name"] = df_results_st["kfold"].apply(lambda fold: fold.split("_")[1])
df_results_st["kfold"] = df_results_st["kfold"].apply(lambda fold: fold.split("_")[0])
# %%
df_arg_str = (
    df_results_st
    .drop(columns=["kfold"])
    .groupby("model_name")
    .agg(["mean", "std"])
    .round(3)
)
# %%
df_arg_str
# %% [markdown]
# Experiment 3: Test on cross-validation using 5 folds.
# Predict on premises, then on conclusion
# %%
X_pre = pd.get_dummies(df_arg_features[[c for c in cw_cols if c != "conclusion_cw_platinum"]]).astype(int)
X_c = pd.get_dummies(df_arg_features[[c for c in cw_cols if c == "conclusion_cw_platinum"]]).astype(int)
# %%
def kfoldres_pc(X_pre, X_c, y):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    models = {
        "lgr": [LogisticRegression(max_iter=1000), LogisticRegression(max_iter=1000)],
        "rforest": [RandomForestClassifier(random_state=0), RandomForestClassifier(random_state=0)],
        "svm": [SVC(random_state=0), SVC(random_state=0)],
        "xgb": [XGBClassifier(use_label_encoder=False,
                              eval_metric="logloss",
                              random_state=0),
                XGBClassifier(use_label_encoder=False,
                              eval_metric="logloss",
                              random_state=0)]
    }

    results = {}

    for split_idx, (train_idx, test_idx) in enumerate(kf.split(X_pre, y)):
        X_train_pre, X_test_pre = X_pre.iloc[train_idx], X_pre.iloc[test_idx]
        X_train_c, X_test_c = X_c.iloc[train_idx], X_c.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_pre = X_train_pre.reset_index(drop=True)
        X_train_c = X_train_c.reset_index(drop=True)
        X_test_pre = X_test_pre.reset_index(drop=True)
        X_test_c = X_test_c.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)


        for model_name, (model_pre, model_c) in models.items():
            model_pre.fit(X_train_pre, y_train)
            preds_pre_test = model_pre.predict(X_test_pre)
            p_pre, r_pre, f1_pre, _ = precision_recall_fscore_support(y_test, preds_pre_test, average="macro")
            results[f"kf={split_idx}_{model_name}_pre"] = {}
            results[f"kf={split_idx}_{model_name}_pre"]["p"] = p_pre
            results[f"kf={split_idx}_{model_name}_pre"]["r"] = r_pre
            results[f"kf={split_idx}_{model_name}_pre"]["f1"] = f1_pre

            preds_pre_train = model_pre.predict(X_train_pre)

            model_c.fit(pd.concat([pd.DataFrame(preds_pre_train), X_train_c],
                                  axis=1,
                                  ignore_index=True),
                        y_train)
            preds_c = model_c.predict(pd.concat([pd.DataFrame(preds_pre_test), X_test_c],
                                                axis=1,
                                                ignore_index=True))

            p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_test, preds_c, average="macro")
            results[f"kf={split_idx}_{model_name}_c"] = {}
            results[f"kf={split_idx}_{model_name}_c"]["p"] = p_c
            results[f"kf={split_idx}_{model_name}_c"]["r"] = r_c
            results[f"kf={split_idx}_{model_name}_c"]["f1"] = f1_c
    
    return results
# %%
results_pc = kfoldres_pc(X_pre, X_c, y)
df_results_pc = pd.DataFrame(results_pc).T
# %%
df_results_pc = df_results_pc.reset_index().rename(columns={"index": "kfold"})
df_results_pc["arg_comp"] = df_results_pc["kfold"].apply(lambda fold: fold.split("_")[2])
df_results_pc["model_name"] = df_results_pc["kfold"].apply(lambda fold: fold.split("_")[1])
df_results_pc["kfold"] = df_results_pc["kfold"].apply(lambda fold: fold.split("_")[0])
# %%
df_p_and_c = (
    df_results_pc
    .drop(columns=["kfold"])
    .groupby(["arg_comp", "model_name"])
    .agg(["mean", "std"])
    .round(3)
)
# %%
df_p_and_c
# %% [markdown]
# Experiment 4: Predict on premise, then on conclusion CW
# %%
X_pre_cw = pd.get_dummies(df_arg_features[[c for c in df_arg_features.drop(columns=hate_cols).columns if "conclusion" not in c]]).astype(int)
X_c_cw = pd.get_dummies(df_arg_features[[c for c in df_arg_features.drop(columns=hate_cols).columns if "conclusion" in c]]).astype(int)
# %%
results_pc_cw = kfoldres_pc(X_pre_cw, X_c_cw, y)
df_results_pc_cw = pd.DataFrame(results_pc_cw).T
# %%
df_results_pc_cw = df_results_pc_cw.reset_index().rename(columns={"index": "kfold"})
df_results_pc_cw["arg_comp"] = df_results_pc_cw["kfold"].apply(lambda fold: fold.split("_")[2])
df_results_pc_cw["model_name"] = df_results_pc_cw["kfold"].apply(lambda fold: fold.split("_")[1])
df_results_pc_cw["kfold"] = df_results_pc_cw["kfold"].apply(lambda fold: fold.split("_")[0])
# %%
df_p_and_c_cw = (
    df_results_pc_cw
    .drop(columns=["kfold"])
    .groupby(["arg_comp", "model_name"])
    .agg(["mean", "std"])
    .round(3)
)
# %%
df_p_and_c_cw
# %% [markdown]
# Experiment 5: Encoding using hate labels per argument component.
# %%
cw_cols = [f"{c}_cw_platinum" for c in columns]
hate_cols = [f"{c}_hate" for c in columns]
# %%
X_cw_h = df_arg_features[[
    col for col in df_arg_features.columns if (("NFS" not in col) and
                                               ("UFS" not in col) and
                                               ("CFS" not in col))
]]
# %%
results_cw_h = kfoldres(X_cw_h, y)
df_results_cw_h = pd.DataFrame(results_cw_h).T
# %%
df_results_cw_h = df_results_cw_h.reset_index().rename(columns={"index": "kfold"})
# %%
df_results_cw_h["model_name"] = df_results_cw_h["kfold"].apply(lambda fold: fold.split("_")[1])
df_results_cw_h["kfold"] = df_results_cw_h["kfold"].apply(lambda fold: fold.split("_")[0])
# %%
df_arg_str_hate = (
    df_results_cw_h
    .drop(columns=["kfold"])
    .groupby("model_name")
    .agg(["mean", "std"])
    .round(3)
)
# %%
df_arg_str_hate
# %%
# %%
X_str_cw_h = df_arg_features
# %%
results_str_cw_h = kfoldres(X_str_cw_h, y)
df_results_str_cw_h = pd.DataFrame(results_str_cw_h).T
# %%
df_results_str_cw_h = df_results_str_cw_h.reset_index().rename(columns={"index": "kfold"})
# %%
df_results_str_cw_h["model_name"] = df_results_str_cw_h["kfold"].apply(lambda fold: fold.split("_")[1])
df_results_str_cw_h["kfold"] = df_results_str_cw_h["kfold"].apply(lambda fold: fold.split("_")[0])
# %%
df_arg_str_hate_cw = (
    df_results_str_cw_h
    .drop(columns=["kfold"])
    .groupby("model_name")
    .agg(["mean", "std"])
    .round(3)
)
# %%
df_arg_str_hate_cw
# %%
# %%
df_combined = pd.concat([
    pd.concat(
        [
            df_arg_str,
            df_arg_str_cw,
            df_arg_str_hate,
            df_arg_str_hate_cw
        ],
        keys=["arg-str", "arg-str-cw", "arg-str-hs", "arg-str-hs-cw"],
        names=["setting", "model_name"]
    ), (
    df_p_and_c
        .rename_axis(index=["setting", "model_name"])
        .rename(index={"c": "arg-str-c-given-p",
                       "pre": "arg-str-p"},
                level="setting")
    ),
    (
    df_p_and_c_cw
        .rename_axis(index=["setting", "model_name"])
        .rename(index={"c": "arg-str-c-given-p-cw",
                       "pre": "arg-str-p-cw"},
                level="setting")
    )
])
# %%
df_combined.index
# %%
df_combined_pm = pd.DataFrame({
    col: df_combined[col]["mean"].map("{:.3f}".format)
         + " ± "
         + df_combined[col]["std"].map("{:.3f}".format)
    for col in df_combined.columns.levels[0]
})
# %%
print(df_combined_pm.loc[[
    "arg-str",
    "arg-str-p",
    "arg-str-c-given-p",
    "arg-str-cw",
    "arg-str-p-cw",
    "arg-str-c-given-p-cw",
    "arg-str-hs",
    "arg-str-hs-cw"
]].to_latex())
# %%