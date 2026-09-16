"""Step 7. Table 2: within- vs between-country dyads (full KODIS-EN corpus).

Sample: human-human dyads with a completed dialogue (joint_points present)
-> N = 2,076, matching the paper.

Match = "within" if b_country == s_country else "between"
(as in the paper; dyads with a missing country label count as "between").

Tests
  Outcome (Resolution vs Impasse): chi-square test of independence.
  points, SVI_{instrumental,self,process,relationship}:
      2 x 2 mixed ANOVA, Role (buyer/seller, within dyad) x Match (between).
      Each DV uses dyads with both buyer and seller values present.

The mixed ANOVA is implemented directly (split-plot sums of squares) so no
extra package is needed; it reproduces every p-value reported in Table 2.
"""
import numpy as np
import pandas as pd
from scipy import stats
from common import KODIS_EN, RESULTS, ensure_dirs

DVS = ["points", "SVI_instrumental", "SVI_self", "SVI_process", "SVI_relationship"]


def mixed_anova_2x2(b, s, group):
    """Role (b vs s, within) x group (between). Returns dict of effects."""
    b, s, group = np.asarray(b, float), np.asarray(s, float), np.asarray(group)
    n, levels = len(b), np.unique(group)
    a = len(levels)
    subj_mean, diff = (b + s) / 2, b - s
    df_err = n - a

    # Between-dyad effect (Match): one-way ANOVA on dyad means (x2 for 2 obs per dyad)
    gm = subj_mean.mean()
    ss_match = sum(2 * (group == g).sum() * (subj_mean[group == g].mean() - gm) ** 2 for g in levels)
    ss_err_between = sum(2 * ((subj_mean[group == g] - subj_mean[group == g].mean()) ** 2).sum()
                         for g in levels)

    # Within-dyad effects (Role, Role x Match): based on buyer-seller differences
    ss_role = n * diff.mean() ** 2 / 2
    ss_inter = sum((group == g).sum() * (diff[group == g].mean() - diff.mean()) ** 2 / 2 for g in levels)
    ss_err_within = sum(((diff[group == g] - diff[group == g].mean()) ** 2).sum() / 2 for g in levels)

    def effect(ss, df1, ss_err):
        F = (ss / df1) / (ss_err / df_err)
        return {"F": F, "df1": df1, "df2": df_err, "p": stats.f.sf(F, df1, df_err),
                "partial_eta2": ss / (ss + ss_err)}

    return {"Match": effect(ss_match, a - 1, ss_err_between),
            "Role": effect(ss_role, 1, ss_err_within),
            "Role x Match": effect(ss_inter, a - 1, ss_err_within)}


if __name__ == "__main__":
    ensure_dirs()

    k = pd.read_csv(KODIS_EN)
    d = k[(k.buyer_is_AI == False) & (k.seller_is_AI == False) & k.joint_points.notna()].copy()  # noqa: E712
    d["Match"] = np.where(d.b_country == d.s_country, "within", "between")
    print(f"N = {len(d)} dyads  ({d.Match.value_counts().to_dict()})")

    # ---- Resolution: chi-square ----
    tab = pd.crosstab(d.Match, d.Outcome)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    rates = pd.crosstab(d.Match, d.Outcome, normalize="index").round(3)
    print(f"\nOutcome chi2({dof}) = {chi2:.3f}, p = {p:.4f}\n{rates}")

    # ---- Mixed ANOVAs ----
    rows = [{"DV": "Outcome (chi-square)", "effect": "Match", "F": chi2, "df1": dof,
             "df2": np.nan, "p": p, "partial_eta2": np.nan, "N": len(d),
             "direction": f"resolution: between {rates.loc['between', 'Resolution']:.1%} "
                          f"vs within {rates.loc['within', 'Resolution']:.1%}"}]
    for dv in DVS:
        x = d[["Match", f"b_{dv}", f"s_{dv}"]].dropna()
        res = mixed_anova_2x2(x[f"b_{dv}"], x[f"s_{dv}"], x.Match)
        mb, ms = x[f"b_{dv}"].mean(), x[f"s_{dv}"].mean()
        dyad = (x[f"b_{dv}"] + x[f"s_{dv}"]) / 2
        mw, mbtw = dyad[x.Match == "within"].mean(), dyad[x.Match == "between"].mean()
        directions = {"Role": f"Buyer {mb:.2f} vs Seller {ms:.2f}",
                      "Match": f"Between {mbtw:.2f} vs Within {mw:.2f}",
                      "Role x Match": ""}
        for eff, r in res.items():
            rows.append({"DV": dv, "effect": eff, **r, "N": len(x), "direction": directions[eff]})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "table2_within_between.csv", index=False)
    show = out.copy()
    show[["F", "p", "partial_eta2"]] = show[["F", "p", "partial_eta2"]].round(4)
    print("\n" + show.to_string(index=False))
