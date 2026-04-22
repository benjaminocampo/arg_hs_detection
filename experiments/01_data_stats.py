# %%
import pandas as pd

df = pd.read_csv("../data/wsf_arg_plus_per_message.csv")
# %%
idx_args = df.loc[(df["is_argument"] == "yes"), "idx"]
# %%
df_claim = pd.read_csv("../data/wsf_arg_plus_per_claim.csv")
# %% [markdown]
# ### Label Distribution in the intersection of Argumentation, Hate Speech, and Check-worthiness
# %%
df_claim_p = df_claim[df_claim["claim_idx"].apply(lambda id: "premise" in id)]
df_claim_c = df_claim[df_claim["claim_idx"].apply(lambda id: "conclusion" in id)]
# %%
pd.crosstab(
    df_claim_p.loc[df_claim_p["concat_hate"] == 1, "claim_hate"],
    df_claim_p.loc[df_claim_p["concat_hate"] == 1, "claim_cw_platinum"], margins=True
).T.loc[["NFS", "UFS", "CFS", "All"]]
# %%
pd.crosstab(
    df_claim_c.loc[df_claim_c["concat_hate"] == 1, "claim_hate"],
    df_claim_c.loc[df_claim_c["concat_hate"] == 1, "claim_cw_platinum"], margins=True
).T.loc[["NFS", "UFS", "CFS", "All"]]
# %%
df_claim_p_non_hs = df_claim_p[(df_claim_p["concat_hate"] == 0) &
                                df_claim_p["claim_idx"].apply(lambda id: id.split("_")[0] in idx_args.astype(str).tolist())]

df_claim_c_non_hs = df_claim_c[(df_claim_c["concat_hate"] == 0) &
                                df_claim_c["claim_idx"].apply(lambda id: id.split("_")[0] in idx_args.astype(str).tolist())]
# %%
df_claim_p_non_hs["claim_cw_platinum"].value_counts().loc[["NFS", "UFS", "CFS"]]
# %%
df_claim_p_non_hs["claim_cw_platinum"].value_counts().loc[["NFS", "UFS", "CFS"]].sum()
# %%
df_claim_c_non_hs["claim_cw_platinum"].value_counts().loc[["NFS", "UFS", "CFS"]]
# %%
df_claim_c_non_hs["claim_cw_platinum"].value_counts().loc[["NFS", "UFS", "CFS"]].sum()
# %%
(
    df_claim_p_non_hs["claim_cw_platinum"].value_counts().loc[["NFS", "UFS", "CFS"]].sum() +
    df_claim_c_non_hs["claim_cw_platinum"].value_counts().loc[["NFS", "UFS", "CFS"]].sum()
)
# %% [markdown]
# ### Data Statistics per message
# %%
df.loc[df["is_argument"] == "yes", "concat_hate"].value_counts()
# %%
df_claim_arg = df_claim[
    (df_claim["claim_idx"].apply(lambda id: id.split("_")[0] in idx_args.astype(str).tolist()))
]
# %%
df_claim_arg.loc[df_claim_arg["concat_hate"] == 1, "claim_idx"].apply(lambda claim: "premise" in claim).sum()
# %%
df_claim_arg.loc[df_claim_arg["concat_hate"] == 1, "claim_idx"].apply(lambda claim: "conclusion" in claim).sum()
# %%
df_claim_arg.loc[df_claim_arg["concat_hate"] == 0, "claim_idx"].apply(lambda claim: "premise" in claim).sum()
# %%
df_claim_arg.loc[df_claim_arg["concat_hate"] == 0, "claim_idx"].apply(lambda claim: "conclusion" in claim).sum()
# %%
len(df_claim_arg.loc[df_claim_arg["concat_hate"] == 1, "claim_idx"])
# %%
len(df_claim_arg.loc[df_claim_arg["concat_hate"] == 0, "claim_idx"])
# %%
p_cols = [
    "premise0",
    "premise1",
    "premise2",
    "premise3",
    "premise4",
    "premise5",
]
c_col = ["conclusion"]
# %%
(
    df
    .loc[(df["is_argument"] == "yes") &
         (df["concat_hate"] == 1),
         [f"{c}_cw_platinum" for c in p_cols]]
    .apply(lambda row: row.notna())
    .sum(axis=1)
).describe().round(3)
# %%
(
    df
    .loc[(df["is_argument"] == "yes") &
         (df["concat_hate"] == 0),
         [f"{c}_cw_platinum" for c in p_cols]]
    .apply(lambda row: row.notna())
    .sum(axis=1)
).describe().round(3)
# %%
(
    df
    .loc[(df["is_argument"] == "yes") &
         (df["concat_hate"] == 1),
         [f"{c}_cw_platinum" for c in p_cols + c_col]]
    .apply(lambda row: row.notna())
    .sum(axis=1)
).describe().round(3)
# %%
(
    df
    .loc[(df["is_argument"] == "yes") &
         (df["concat_hate"] == 0),
         [f"{c}_cw_platinum" for c in p_cols + c_col]]
    .apply(lambda row: row.notna())
    .sum(axis=1)
).describe().round(3)
# %%
