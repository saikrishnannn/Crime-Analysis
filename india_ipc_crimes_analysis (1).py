# ============================================================
#   INDIA DISTRICT-WISE IPC CRIMES ANALYSIS  (2001 – 2012)
#   Source : NCRB — National Crime Records Bureau, Govt. of India
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")



# ── GLOBAL STYLE ─────────────────────────────────────────────
PALETTE = {
    "primary"   : "#1B4F72",   # deep navy
    "secondary" : "#2E86C1",   # medium blue
    "accent"    : "#E74C3C",   # vivid red (danger / highlight)
    "warm"      : "#E67E22",   # amber / orange
    "teal"      : "#1ABC9C",   # teal green
    "purple"    : "#8E44AD",   # violet
    "light"     : "#ECF0F1",   # off-white surface
    "dark"      : "#1C2833",   # near-black bg
    "muted"     : "#7F8C8D",   # grey text
    "gridline"  : "#D5D8DC",   # faint grid
}

SEQUENTIAL = ["#D6EAF8", "#85C1E9", "#2E86C1", "#1B4F72", "#0B2641"]

plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#FDFEFE",
    "axes.edgecolor"    : PALETTE["gridline"],
    "axes.linewidth"    : 0.8,
    "axes.grid"         : True,
    "grid.color"        : PALETTE["gridline"],
    "grid.linewidth"    : 0.5,
    "grid.alpha"        : 0.6,
    "xtick.color"       : PALETTE["muted"],
    "ytick.color"       : PALETTE["muted"],
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "font.family"       : "DejaVu Sans",
    "text.color"        : PALETTE["dark"],
})

def style_axis(ax, title, xlabel="", ylabel="", subtitle=""):
    """Apply consistent professional axis styling."""
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=PALETTE["primary"], pad=14, loc="left")
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes,
                fontsize=9, color=PALETTE["muted"], va="bottom")
    ax.set_xlabel(xlabel, fontsize=9, color=PALETTE["muted"], labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9, color=PALETTE["muted"], labelpad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)

def save_fig(fig, name):
    fig.tight_layout()
    plt.show()


# ============================================================
# PHASE 1 — LOAD & CLEAN
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 1 : DATA LOADING & CLEANING")
print("=" * 60)

df = pd.read_csv("data1.csv")
df.columns = df.columns.str.strip()

df.rename(columns={
    df.columns[0]: "State",
    df.columns[1]: "District",
    df.columns[2]: "Year",
}, inplace=True)

# Remove state-level TOTAL aggregate rows
df = df[df["District"].str.upper() != "TOTAL"].copy()

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
crime_cols = df.columns[3:]
for col in crime_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Rename the canonical total column
if "TOTAL IPC CRIMES" in df.columns:
    df.rename(columns={"TOTAL IPC CRIMES": "TOTAL_IPC"}, inplace=True)
else:
    df["TOTAL_IPC"] = df[crime_cols].sum(axis=1)

df.reset_index(drop=True, inplace=True)

print(f"  Rows after cleaning  : {len(df):,}")
print(f"  Years covered        : {sorted(df['Year'].dropna().unique().astype(int))}")
print(f"  States               : {df['State'].nunique()}")
print(f"  Districts            : {df['District'].nunique()}")
print(f"  Total IPC (sample)   :\n{df['TOTAL_IPC'].describe().round(1)}")


# ============================================================
# PHASE 2 — EXPLORATORY DATA ANALYSIS  (5 Objectives)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 2 : EXPLORATORY DATA ANALYSIS")
print("=" * 60)

CRIME_CATS = [
    "MURDER", "RAPE", "KIDNAPPING & ABDUCTION",
    "ROBBERY", "BURGLARY", "THEFT", "RIOTS",
    "CHEATING", "DOWRY DEATHS",
    "CRUELTY BY HUSBAND OR HIS RELATIVES",
    "CAUSING DEATH BY NEGLIGENCE",
]
# Keep only columns that actually exist in this dataset
CRIME_CATS = [c for c in CRIME_CATS if c in df.columns]

# ── Objective 1 : National Trend ──────────────────────────
national = (df.groupby("Year")["TOTAL_IPC"]
              .sum()
              .reset_index()
              .rename(columns={"TOTAL_IPC": "Total_Crimes"}))
national["YoY_%"] = national["Total_Crimes"].pct_change().mul(100).round(2)

print("\n── Obj 1 : National IPC Crime Trend ──")
print(national.to_string(index=False))

# ── Objective 2 : Top / Bottom States ────────────────────
state_total = (df.groupby("State")["TOTAL_IPC"]
                 .sum()
                 .reset_index()
                 .rename(columns={"TOTAL_IPC": "Total_Crimes"})
                 .sort_values("Total_Crimes", ascending=False)
                 .reset_index(drop=True))

print("\n── Obj 2 : Top 5 States ──")
print(state_total.head(5).to_string(index=False))
print("   Bottom 5 :")
print(state_total.tail(5).to_string(index=False))

# ── Objective 3 : Crime-Type Growth 2001→2012 ────────────
y2001 = df[df["Year"] == 2001][CRIME_CATS].sum()
y2012 = df[df["Year"] == 2012][CRIME_CATS].sum()

growth = pd.DataFrame({
    "Crime_Type"  : CRIME_CATS,
    "Count_2001"  : y2001.values,
    "Count_2012"  : y2012.values,
})
growth["Growth_%"] = ((growth["Count_2012"] - growth["Count_2001"])
                      / growth["Count_2001"] * 100).round(2)
growth.sort_values("Growth_%", ascending=False, inplace=True)

print("\n── Obj 3 : Crime Growth 2001 → 2012 ──")
print(growth.to_string(index=False))

# ── Objective 4 : YoY already in national df ─────────────
print("\n── Obj 4 : Year-over-Year Growth Rate ──")
print(national.to_string(index=False))

# ── Objective 5 : Hotspot Districts ──────────────────────
district_total = (df.groupby(["State", "District"])["TOTAL_IPC"]
                    .sum()
                    .reset_index()
                    .rename(columns={"TOTAL_IPC": "Total_Crimes"})
                    .sort_values("Total_Crimes", ascending=False))

print("\n── Obj 5 : Top 10 Hotspot Districts ──")
print(district_total.head(10).to_string(index=False))


# ============================================================
# PHASE 3 — VISUALISATIONS  (professional restyle)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 3 : VISUALISATIONS")
print("=" * 60)

# ── Plot 1 : National Crime Trend  ───────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

ax.fill_between(national["Year"], national["Total_Crimes"],
                alpha=0.12, color=PALETTE["secondary"])
ax.plot(national["Year"], national["Total_Crimes"],
        color=PALETTE["primary"], linewidth=2.5,
        marker="o", markersize=7, markerfacecolor="white",
        markeredgewidth=2, markeredgecolor=PALETTE["primary"], zorder=5)

for _, row in national.iterrows():
    ax.annotate(f'{int(row["Total_Crimes"]):,}',
                xy=(row["Year"], row["Total_Crimes"]),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=7.5, color=PALETTE["muted"])

ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
ax.set_xticks(national["Year"])
style_axis(ax,
           title="National IPC Crime Trend  |  2001 – 2012",
           subtitle="Total cognisable offences reported across all districts",
           xlabel="Year", ylabel="Total IPC Crimes Reported")

save_fig(fig, "01_national_trend")

# ── Plot 2 : Top 10 States — Horizontal Bar  ─────────────
top10 = state_total.head(10).iloc[::-1].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 6))
colors_bar = [PALETTE["accent"] if i == len(top10) - 1
              else PALETTE["secondary"] for i in range(len(top10))]
bars = ax.barh(top10["State"], top10["Total_Crimes"],
               color=colors_bar, height=0.6, edgecolor="white",
               linewidth=0.5)

for bar in bars:
    w = bar.get_width()
    ax.text(w + top10["Total_Crimes"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{int(w):,}', va="center", fontsize=8,
            color=PALETTE["muted"])

ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.set_xlim(0, top10["Total_Crimes"].max() * 1.15)
style_axis(ax,
           title="Top 10 States by Cumulative IPC Crimes  |  2001 – 2012",
           subtitle="Highlighted bar = highest reporting state",
           xlabel="Total IPC Crimes (Cumulative)", ylabel="")

save_fig(fig, "02_top10_states")

# ── Plot 3 : Crime-Type Growth — Diverging Bar  ──────────
fig, ax = plt.subplots(figsize=(12, 6))

bar_colors = [PALETTE["accent"] if x > 0 else PALETTE["teal"]
              for x in growth["Growth_%"]]
ax.barh(growth["Crime_Type"], growth["Growth_%"],
        color=bar_colors, height=0.55, edgecolor="white")

ax.axvline(0, color=PALETTE["dark"], linewidth=0.8)
for i, (_, row) in enumerate(growth.iterrows()):
    offset = 1 if row["Growth_%"] >= 0 else -1
    ha = "left" if row["Growth_%"] >= 0 else "right"
    ax.text(row["Growth_%"] + offset, i,
            f'{row["Growth_%"]:+.1f}%', va="center",
            ha=ha, fontsize=8, color=PALETTE["muted"])

ax.set_xlabel("Growth Rate (%)", fontsize=9, color=PALETTE["muted"])
style_axis(ax,
           title="Crime Category Growth Rate  |  2001 → 2012",
           subtitle="Red = increase  |  Teal = decrease")

save_fig(fig, "03_crime_growth")

# ── Plot 4 : YoY Growth Rate — Column Chart  ─────────────
yoy = national.dropna(subset=["YoY_%"])

fig, ax = plt.subplots(figsize=(11, 5))
cols = [PALETTE["accent"] if v >= 0 else PALETTE["teal"]
        for v in yoy["YoY_%"]]
ax.bar(yoy["Year"], yoy["YoY_%"], color=cols,
       width=0.6, edgecolor="white", linewidth=0.5)
ax.axhline(0, color=PALETTE["dark"], linewidth=0.9)

for _, row in yoy.iterrows():
    va = "bottom" if row["YoY_%"] >= 0 else "top"
    offset = 0.1 if row["YoY_%"] >= 0 else -0.1
    ax.text(row["Year"], row["YoY_%"] + offset,
            f'{row["YoY_%"]:+.1f}%', ha="center",
            va=va, fontsize=8, color=PALETTE["muted"])

ax.set_xticks(yoy["Year"])
style_axis(ax,
           title="Year-over-Year National Crime Growth Rate  |  2002 – 2012",
           subtitle="Positive = more crimes than previous year  |  Negative = decline",
           xlabel="Year", ylabel="Growth Rate (%)")

save_fig(fig, "04_yoy_growth")

# ── Plot 5 : Heatmap — Top 15 States × Year  ─────────────
top15_states = state_total.head(15)["State"].tolist()
hmap_data = (df[df["State"].isin(top15_states)]
               .groupby(["State", "Year"])["TOTAL_IPC"]
               .sum()
               .reset_index())
hmap_pivot = hmap_data.pivot(index="State", columns="Year",
                              values="TOTAL_IPC")

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(
    hmap_pivot,
    cmap=sns.color_palette("Blues", as_cmap=True),
    linewidths=0.4, linecolor="white",
    annot=True, fmt=".0f", annot_kws={"size": 6.5},
    cbar_kws={"shrink": 0.7, "label": "Total IPC Crimes"},
    ax=ax,
)
ax.set_title("State × Year Crime Heatmap  |  Top 15 States  (2001 – 2012)",
             fontsize=13, fontweight="bold",
             color=PALETTE["primary"], pad=14, loc="left")
ax.set_xlabel("Year", fontsize=9, color=PALETTE["muted"])
ax.set_ylabel("", fontsize=9)
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)

save_fig(fig, "05_heatmap_states")


# ============================================================
# PHASE 4 — LINEAR REGRESSION  (national trend)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 4 : LINEAR REGRESSION MODEL")
print("=" * 60)

X   = national[["Year"]]
y   = national["Total_Crimes"]

model_lr = LinearRegression()
model_lr.fit(X, y)

y_pred = model_lr.predict(X)
cv_r2  = cross_val_score(LinearRegression(), X, y, cv=4, scoring="r2")

mae  = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2   = r2_score(y, y_pred)

print(f"\n  Equation  :  Crimes = {model_lr.coef_[0]:,.0f} × Year"
      f" + ({model_lr.intercept_:,.0f})")
print(f"  Annual rise (slope)     : ~{model_lr.coef_[0]:,.0f} crimes/year")
print(f"  MAE                     : {mae:,.0f}")
print(f"  RMSE                    : {rmse:,.0f}")
print(f"  R²  (full data)         : {r2:.4f}  ({r2:.1%} variance explained)")
print(f"  Mean CV R²  (k=4)       : {cv_r2.mean():.4f}")

future_lr = pd.DataFrame({"Year": range(2013, 2020)})
future_lr["Predicted"] = model_lr.predict(future_lr[["Year"]]).astype(int)

print("\n  ── Linear Forecast 2013 – 2019 ──")
print(future_lr.to_string(index=False))

# Actuals table
print("\n  ── Actual vs Predicted ──")
print(f"  {'Year':<6} {'Actual':>12} {'Predicted':>12} {'Error':>10}")
print("  " + "-" * 44)
for yr, act, pred in zip(X["Year"], y, y_pred):
    print(f"  {int(yr):<6} {int(act):>12,} {int(pred):>12,} {int(act-pred):>+10,}")


# ============================================================
# PHASE 5 — POLYNOMIAL REGRESSION  (new / upgraded prediction)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 5 : POLYNOMIAL REGRESSION MODEL  (degree = 2)")
print("=" * 60)

poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X, y)

y_poly     = poly_model.predict(X)
cv_poly    = cross_val_score(
    make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    X, y, cv=4, scoring="r2")

mae_p  = mean_absolute_error(y, y_poly)
rmse_p = np.sqrt(mean_squared_error(y, y_poly))
r2_p   = r2_score(y, y_poly)

print(f"\n  MAE   (polynomial) : {mae_p:,.0f}")
print(f"  RMSE  (polynomial) : {rmse_p:,.0f}")
print(f"  R²    (polynomial) : {r2_p:.4f}")
print(f"  CV R² (polynomial) : {cv_poly.mean():.4f}")

future_poly            = pd.DataFrame({"Year": range(2013, 2020)})
future_poly["Predicted"] = poly_model.predict(future_poly[["Year"]]).astype(int)

print("\n  ── Polynomial Forecast 2013 – 2019 ──")
print(future_poly.to_string(index=False))


# ============================================================
# PHASE 6 — CRIME-TYPE FORECASTING  (individual linear models)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 6 : PER-CRIME-TYPE FORECASTING  (2013 – 2019)")
print("=" * 60)

forecast_rows = []
for crime in CRIME_CATS:
    ct = (df.groupby("Year")[crime].sum().reset_index()
            .rename(columns={crime: "Count"}))
    Xc = ct[["Year"]]
    yc = ct["Count"]
    m  = LinearRegression().fit(Xc, yc)
    for yr in range(2013, 2020):
        pred = max(0, int(m.predict([[yr]])[0]))
        forecast_rows.append({"Crime_Type": crime, "Year": yr,
                               "Forecast": pred})

forecast_df = pd.DataFrame(forecast_rows)
pivot_fc    = forecast_df.pivot(index="Crime_Type",
                                 columns="Year",
                                 values="Forecast")
print(pivot_fc.to_string())

# Export forecasts
forecast_df.to_csv("crime_type_forecast_2013_2019.csv", index=False)
national.to_csv("national_trend.csv", index=False)
growth.to_csv("crime_growth_2001_2012.csv", index=False)
print("\n  ✔  CSVs saved alongside script")


# ============================================================
# PHASE 7 — PREDICTION VISUALISATIONS  (3 new plots)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 7 : PREDICTION VISUALISATIONS")
print("=" * 60)

all_years      = pd.DataFrame({"Year": range(2001, 2020)})
pred_lr_all    = model_lr.predict(all_years[["Year"]])
pred_poly_all  = poly_model.predict(all_years[["Year"]])

# ── Plot 6 : Linear vs Polynomial Regression + Forecast ──
fig, ax = plt.subplots(figsize=(13, 6))

# Shaded forecast zone
ax.axvspan(2012.5, 2019.5, color=PALETTE["light"], alpha=0.7,
           label="Forecast zone (2013–2019)", zorder=0)

# Actual data
ax.scatter(X["Year"], y, color=PALETTE["primary"], s=80,
           zorder=6, label="Actual data  (2001–2012)")

# Regression lines
ax.plot(all_years["Year"], pred_lr_all,
        color=PALETTE["warm"], linewidth=2, linestyle="--",
        label=f"Linear  (R² = {r2:.3f})", zorder=4)
ax.plot(all_years["Year"], pred_poly_all,
        color=PALETTE["accent"], linewidth=2.2,
        label=f"Polynomial deg-2  (R² = {r2_p:.3f})", zorder=5)

# Forecast dots
for _, row in future_poly.iterrows():
    ax.scatter(row["Year"], row["Predicted"],
               color=PALETTE["accent"], s=65,
               marker="D", zorder=7)
    ax.annotate(f'{row["Predicted"]:,}',
                xy=(row["Year"], row["Predicted"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=7, color=PALETTE["accent"])

ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.set_xticks(range(2001, 2020))
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=8.5, framealpha=0.85)

style_axis(ax,
           title="Crime Trend Regression & Forecast  |  2001 – 2019",
           subtitle="Linear vs Polynomial (degree 2) regression  |  Diamonds = projected values",
           xlabel="Year", ylabel="Total IPC Crimes Reported")

save_fig(fig, "06_regression_forecast")

# ── Plot 7 : Per-Crime-Type Forecast  2013 – 2019  ────────
fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharey=False)
axes = axes.flatten()

crime_colors = [PALETTE["primary"], PALETTE["secondary"],
                PALETTE["accent"],  PALETTE["warm"],
                PALETTE["teal"],    PALETTE["purple"],
                "#C0392B", "#117A65", "#784212",
                "#1A5276", "#6C3483"]

for idx, crime in enumerate(CRIME_CATS):
    ax = axes[idx]
    ct = (df.groupby("Year")[crime].sum().reset_index()
            .rename(columns={crime: "Count"}))
    fc = forecast_df[forecast_df["Crime_Type"] == crime]
    c  = crime_colors[idx % len(crime_colors)]

    ax.plot(ct["Year"], ct["Count"],
            color=c, linewidth=2, marker="o", markersize=5,
            markerfacecolor="white", markeredgewidth=1.5,
            markeredgecolor=c, label="Actual")
    ax.plot(fc["Year"], fc["Forecast"],
            color=c, linewidth=1.8, linestyle="--",
            marker="D", markersize=4, label="Forecast")
    ax.axvline(2012.5, color=PALETTE["muted"],
               linewidth=0.7, linestyle=":")

    ax.set_title(crime.title(), fontsize=8.5,
                 fontweight="bold", color=PALETTE["primary"], loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7, length=0)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.set_xticks([2001, 2006, 2012, 2019])
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=6.5, framealpha=0.7, loc="upper left")

# Hide unused subplot slots
for j in range(len(CRIME_CATS), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Per-Crime-Type Trend & Forecast  |  2001 – 2019\n"
    "Solid = actual  |  Dashed = linear forecast  |  Dotted line = 2012 cutoff",
    fontsize=12, fontweight="bold", color=PALETTE["primary"], y=1.01)
save_fig(fig, "07_per_crime_forecast")

# ── Plot 8 : Model Accuracy Comparison Dashboard  ─────────
metrics = {
    "Model"  : ["Linear Regression", "Polynomial (deg 2)"],
    "MAE"    : [mae,   mae_p],
    "RMSE"   : [rmse,  rmse_p],
    "R²"     : [r2,    r2_p],
    "CV R²"  : [cv_r2.mean(), cv_poly.mean()],
}
metrics_df = pd.DataFrame(metrics)

fig = plt.figure(figsize=(14, 6))
gs  = GridSpec(1, 3, figure=fig, wspace=0.38)

# Bar: MAE comparison
ax1 = fig.add_subplot(gs[0])
bars = ax1.bar(metrics_df["Model"], metrics_df["MAE"],
               color=[PALETTE["secondary"], PALETTE["accent"]],
               width=0.45, edgecolor="white")
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 500,
             f'{int(bar.get_height()):,}', ha="center",
             fontsize=8.5, color=PALETTE["muted"])
style_axis(ax1, title="MAE  (lower = better)", ylabel="Mean Absolute Error")
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax1.set_xticklabels(metrics_df["Model"], fontsize=8)

# Bar: R² comparison
ax2 = fig.add_subplot(gs[1])
bars2 = ax2.bar(metrics_df["Model"], metrics_df["R²"],
                color=[PALETTE["secondary"], PALETTE["accent"]],
                width=0.45, edgecolor="white")
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005,
             f'{bar.get_height():.3f}', ha="center",
             fontsize=8.5, color=PALETTE["muted"])
ax2.set_ylim(0, 1)
style_axis(ax2, title="R²  (higher = better)", ylabel="R² Score")
ax2.set_xticklabels(metrics_df["Model"], fontsize=8)

# Table: full metrics
ax3 = fig.add_subplot(gs[2])
ax3.axis("off")
tbl = ax3.table(
    cellText  = [[m,
                  f'{int(a):,}',
                  f'{int(r):,}',
                  f'{r2v:.4f}',
                  f'{cv:.4f}']
                 for m, a, r, r2v, cv in zip(
                     metrics_df["Model"], metrics_df["MAE"],
                     metrics_df["RMSE"], metrics_df["R²"],
                     metrics_df["CV R²"])],
    colLabels = ["Model", "MAE", "RMSE", "R²", "CV R²"],
    cellLoc   = "center",
    loc       = "center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 2.2)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(PALETTE["gridline"])
    if row == 0:
        cell.set_facecolor(PALETTE["primary"])
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 1:
        cell.set_facecolor("#EBF5FB")
    else:
        cell.set_facecolor("white")

ax3.set_title("Full Accuracy Summary", fontsize=10,
              fontweight="bold", color=PALETTE["primary"],
              pad=16, loc="left")

fig.suptitle("Model Accuracy Comparison  |  Linear vs Polynomial Regression",
             fontsize=12, fontweight="bold",
             color=PALETTE["primary"], y=1.02)
save_fig(fig, "08_model_accuracy")




# ============================================================
# PHASE 8 — PER-CAPITA NORMALIZATION  (Census 2011 population)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 8 : PER-CAPITA NORMALIZATION")
print("=" * 60)

# 2011 Census population (in actual numbers) for each state/UT
# Source: Census of India 2011 — Office of the Registrar General
POPULATION_2011 = {
    "ANDHRA PRADESH"                  : 84580777,
    "ARUNACHAL PRADESH"               : 1383727,
    "ASSAM"                           : 31205576,
    "BIHAR"                           : 104099452,
    "CHHATTISGARH"                    : 25545198,
    "GOA"                             : 1458545,
    "GUJARAT"                         : 60439692,
    "HARYANA"                         : 25351462,
    "HIMACHAL PRADESH"                : 6864602,
    "JAMMU & KASHMIR"                 : 12541302,
    "JHARKHAND"                       : 32988134,
    "KARNATAKA"                       : 61095297,
    "KERALA"                          : 33406061,
    "MADHYA PRADESH"                  : 72626809,
    "MAHARASHTRA"                     : 112374333,
    "MANIPUR"                         : 2855794,
    "MEGHALAYA"                       : 2966889,
    "MIZORAM"                         : 1097206,
    "NAGALAND"                        : 1978502,
    "ODISHA"                          : 41974218,
    "PUNJAB"                          : 27743338,
    "RAJASTHAN"                       : 68548437,
    "SIKKIM"                          : 610577,
    "TAMIL NADU"                      : 72147030,
    "TELANGANA"                       : 35003674,
    "TRIPURA"                         : 3673917,
    "UTTAR PRADESH"                   : 199812341,
    "UTTARAKHAND"                     : 10086292,
    "WEST BENGAL"                     : 91276115,
    "ANDAMAN & NICOBAR ISLANDS"       : 380581,
    "CHANDIGARH"                      : 1055450,
    "DADRA & NAGAR HAVELI"            : 343709,
    "DAMAN & DIU"                     : 243247,
    "DELHI"                           : 16787941,
    "LAKSHADWEEP"                     : 64473,
    "PUDUCHERRY"                      : 1247953,
}

# Build per-capita dataframe
pop_df = pd.DataFrame([
    {"State": s, "Population": p}
    for s, p in POPULATION_2011.items()
])

# Normalize state names in our data to uppercase for matching
state_total["State_upper"] = state_total["State"].str.upper().str.strip()
pop_df["State_upper"]      = pop_df["State"].str.upper().str.strip()

percapita = state_total.merge(pop_df[["State_upper", "Population"]],
                               on="State_upper", how="left")
percapita["Crimes_per_Lakh"] = (
    percapita["Total_Crimes"] / percapita["Population"] * 100000
).round(2)

# Drop rows where population not found
missing = percapita[percapita["Population"].isna()]["State"].tolist()
if missing:
    print(f"  Note: population not found for: {missing}")

percapita.dropna(subset=["Population"], inplace=True)
percapita.sort_values("Crimes_per_Lakh", ascending=False, inplace=True)
percapita.reset_index(drop=True, inplace=True)

print("\n  Top 10 States — Raw Total vs Per-Lakh Population:")
print(f"  {'State':<35} {'Raw Total':>12} {'Per Lakh':>10} {'Population':>14}")
print("  " + "-" * 75)
for _, row in percapita.head(10).iterrows():
    print(f"  {row['State']:<35} {int(row['Total_Crimes']):>12,} "
          f"{row['Crimes_per_Lakh']:>10.1f} {int(row['Population']):>14,}")

print("\n  Bottom 5 States by Per-Lakh rate:")
for _, row in percapita.tail(5).iterrows():
    print(f"  {row['State']:<35} {row['Crimes_per_Lakh']:>10.1f} per lakh")

# Export
percapita[["State", "Total_Crimes", "Population",
           "Crimes_per_Lakh"]].to_csv("percapita_crime_rates.csv", index=False)
print("\n  \u2714  percapita_crime_rates.csv saved")

# ── Plot 9 (inserted) : Raw vs Per-Capita comparison ─────────
top10_raw       = state_total.head(10)["State"].str.upper().str.strip().tolist()
top10_percapita = percapita.head(10)["State_upper"].tolist()
combined_states = list(dict.fromkeys(top10_raw + top10_percapita))  # unique, order preserved

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left — Raw totals (top 10)
top10_r = state_total.head(10).iloc[::-1]
bar_colors_r = [PALETTE["accent"] if i == 9 else PALETTE["secondary"]
                for i in range(10)]
axes[0].barh(top10_r["State"], top10_r["Total_Crimes"],
             color=bar_colors_r, height=0.6, edgecolor="white")
axes[0].xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
for bar in axes[0].patches:
    w = bar.get_width()
    axes[0].text(w + top10_r["Total_Crimes"].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{int(w):,}', va="center", fontsize=7.5,
                 color=PALETTE["muted"])
style_axis(axes[0],
           title="Top 10 States  |  Raw Crime Total",
           subtitle="2001 – 2012 cumulative",
           xlabel="Total IPC Crimes")

# Right — Per-lakh (top 10)
top10_pc = percapita.head(10).iloc[::-1]
bar_colors_pc = [PALETTE["accent"] if i == 9 else PALETTE["teal"]
                 for i in range(10)]
axes[1].barh(top10_pc["State"], top10_pc["Crimes_per_Lakh"],
             color=bar_colors_pc, height=0.6, edgecolor="white")
for bar in axes[1].patches:
    w = bar.get_width()
    axes[1].text(w + percapita["Crimes_per_Lakh"].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{w:.0f}', va="center", fontsize=7.5,
                 color=PALETTE["muted"])
style_axis(axes[1],
           title="Top 10 States  |  Per Lakh Population",
           subtitle="Normalized using Census 2011 population",
           xlabel="Crimes per 1,00,000 people")

fig.suptitle(
    "Raw vs Per-Capita Crime Rate  |  The ranking changes significantly after normalization",
    fontsize=12, fontweight="bold", color=PALETTE["primary"], y=1.01)
fig.tight_layout()
save_fig(fig, "09_percapita_comparison")

# ============================================================
# PHASE 9 — ANOMALY DETECTION  (z-score flagging)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 9 : ANOMALY DETECTION  (national trend)")
print("=" * 60)

# Z-score on the national yearly totals
national["Z_Score"] = scipy_stats.zscore(national["Total_Crimes"])
THRESHOLD = 1.2   # flag anything beyond ±1.2 SD
national["Anomaly"] = national["Z_Score"].abs() > THRESHOLD

anomalies = national[national["Anomaly"]]
print(f"\n  Threshold  : ±{THRESHOLD} standard deviations")
print(f"  Anomalies detected : {len(anomalies)}")
print("\n  Flagged Years:")
print(f"  {'Year':<8} {'Total_Crimes':>14} {'Z_Score':>10} {'Direction':>12}")
print("  " + "-" * 48)
for _, row in anomalies.iterrows():
    direction = "SPIKE (high)" if row["Z_Score"] > 0 else "DROP  (low)"
    print(f"  {int(row['Year']):<8} {int(row['Total_Crimes']):>14,} "
          f"{row['Z_Score']:>10.3f} {direction:>12}")

# ── Per-crime-type anomaly scan ──────────────────────────────
print("\n  Per-Crime-Type Anomalous Years:")
print(f"  {'Crime Type':<42} {'Year':>6} {'Z':>7}")
print("  " + "-" * 58)
for crime in CRIME_CATS:
    ct = df.groupby("Year")[crime].sum().reset_index()
    ct["Z"] = scipy_stats.zscore(ct[crime])
    spikes = ct[ct["Z"].abs() > THRESHOLD]
    for _, row in spikes.iterrows():
        direction = "+" if row["Z"] > 0 else "-"
        print(f"  {crime:<42} {int(row['Year']):>6} {direction}{abs(row['Z']):.2f}")

# ── Plot 9 : Anomaly Detection Chart ─────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 10))

# Top panel — actual trend with anomaly flags
ax = axes[0]
ax.fill_between(national["Year"], national["Total_Crimes"],
                alpha=0.08, color=PALETTE["secondary"])
ax.plot(national["Year"], national["Total_Crimes"],
        color=PALETTE["primary"], linewidth=2.5,
        marker="o", markersize=7,
        markerfacecolor="white", markeredgewidth=2,
        markeredgecolor=PALETTE["primary"], zorder=4,
        label="Annual total")

# Shade normal band (mean ± threshold × std)
mean_c = national["Total_Crimes"].mean()
std_c  = national["Total_Crimes"].std()
ax.axhspan(mean_c - THRESHOLD * std_c,
           mean_c + THRESHOLD * std_c,
           color=PALETTE["teal"], alpha=0.08,
           label=f"Normal band  (±{THRESHOLD}σ)")
ax.axhline(mean_c, color=PALETTE["teal"],
           linewidth=1.2, linestyle="--", alpha=0.7,
           label=f"Mean  ({int(mean_c):,})")

# Red markers for anomalies
for _, row in anomalies.iterrows():
    ax.scatter(row["Year"], row["Total_Crimes"],
               color=PALETTE["accent"], s=130, zorder=6,
               marker="^" if row["Z_Score"] > 0 else "v")
    va = "bottom" if row["Z_Score"] > 0 else "top"
    offset = 35000 if row["Z_Score"] > 0 else -35000
    ax.annotate(
        f'{int(row["Year"])}  (z={row["Z_Score"]:+.2f})',
        xy=(row["Year"], row["Total_Crimes"]),
        xytext=(0, offset), textcoords="offset points",
        ha="center", fontsize=8,
        color=PALETTE["accent"],
        arrowprops=dict(arrowstyle="-", color=PALETTE["accent"],
                        lw=0.8, alpha=0.6),
    )

ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
ax.set_xticks(national["Year"])
ax.legend(fontsize=8.5, framealpha=0.85, loc="upper left")
style_axis(ax,
           title="Anomaly Detection  |  National IPC Crime Trend  (2001 – 2012)",
           subtitle="Triangles = flagged anomalies  |  Shaded band = normal range",
           xlabel="Year", ylabel="Total IPC Crimes")

# Bottom panel — Z-score bar chart
ax2 = axes[1]
bar_colors = [PALETTE["accent"] if abs(z) > THRESHOLD
              else PALETTE["secondary"]
              for z in national["Z_Score"]]
ax2.bar(national["Year"], national["Z_Score"],
        color=bar_colors, width=0.6, edgecolor="white", linewidth=0.5)
ax2.axhline( THRESHOLD, color=PALETTE["accent"],
             linewidth=1.2, linestyle="--", alpha=0.7,
             label=f"+{THRESHOLD}σ threshold")
ax2.axhline(-THRESHOLD, color=PALETTE["accent"],
             linewidth=1.2, linestyle="--", alpha=0.7,
             label=f"-{THRESHOLD}σ threshold")
ax2.axhline(0, color=PALETTE["dark"], linewidth=0.8)

for _, row in national.iterrows():
    va = "bottom" if row["Z_Score"] >= 0 else "top"
    offset = 0.05 if row["Z_Score"] >= 0 else -0.05
    ax2.text(row["Year"], row["Z_Score"] + offset,
             f'{row["Z_Score"]:+.2f}', ha="center",
             va=va, fontsize=7.5, color=PALETTE["muted"])

ax2.set_xticks(national["Year"])
ax2.legend(fontsize=8.5, framealpha=0.85)
style_axis(ax2,
           title="Z-Score per Year  |  Deviation from Mean",
           subtitle="Red bars = anomalous years outside the threshold",
           xlabel="Year", ylabel="Z-Score (standard deviations)")

fig.tight_layout(pad=3.0)
save_fig(fig, "10_anomaly_detection")


# ============================================================
# PHASE 10 — ARIMA TIME SERIES MODEL
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 10 : ARIMA TIME SERIES FORECASTING")
print("=" * 60)

ts = national.set_index("Year")["Total_Crimes"].astype(float)

# ── Stationarity check (ADF test) ────────────────────────────
adf_result = adfuller(ts, autolag="AIC")
print(f"\n  ADF Stationarity Test")
print(f"  ADF Statistic  : {adf_result[0]:.4f}")
print(f"  p-value        : {adf_result[1]:.4f}")
print(f"  Stationary     : {'YES' if adf_result[1] < 0.05 else 'NO — differencing needed (handled by d=1)'}")

# ── Fit ARIMA(1,1,1) ─────────────────────────────────────────
#   p=1 : one lag of the series
#   d=1 : first-order differencing (makes it stationary)
#   q=1 : one lag of the forecast error
arima_model  = ARIMA(ts, order=(1, 1, 1))
arima_result = arima_model.fit()

print(f"\n  ARIMA(1,1,1) Summary")
print(f"  AIC  : {arima_result.aic:.2f}")
print(f"  BIC  : {arima_result.bic:.2f}")

# In-sample fitted values
fitted = arima_result.fittedvalues

mae_a  = mean_absolute_error(ts.iloc[1:], fitted.iloc[1:])
rmse_a = np.sqrt(mean_squared_error(ts.iloc[1:], fitted.iloc[1:]))
r2_a   = r2_score(ts.iloc[1:], fitted.iloc[1:])

print(f"\n  In-Sample Accuracy (on known data 2002–2012):")
print(f"  MAE   : {mae_a:,.0f}")
print(f"  RMSE  : {rmse_a:,.0f}")
print(f"  R²    : {r2_a:.4f}")

# ── Forecast 2013–2019 with 95% confidence interval ──────────
FORECAST_STEPS = 7
forecast_obj   = arima_result.get_forecast(steps=FORECAST_STEPS)
forecast_mean  = forecast_obj.predicted_mean
forecast_ci    = forecast_obj.conf_int(alpha=0.05)   # 95% CI

forecast_years = list(range(2013, 2013 + FORECAST_STEPS))
forecast_mean.index  = forecast_years
forecast_ci.index    = forecast_years

print(f"\n  ARIMA Forecast  2013 – {2012 + FORECAST_STEPS}  (95% Confidence Interval):")
print(f"  {'Year':<8} {'Forecast':>14} {'Lower 95%':>12} {'Upper 95%':>12}")
print("  " + "-" * 50)
for yr in forecast_years:
    print(f"  {yr:<8} {int(forecast_mean[yr]):>14,} "
          f"{int(forecast_ci.iloc[yr-2013, 0]):>12,} "
          f"{int(forecast_ci.iloc[yr-2013, 1]):>12,}")

# Export ARIMA forecast
arima_df = pd.DataFrame({
    "Year"     : forecast_years,
    "Forecast" : forecast_mean.values.astype(int),
    "Lower_95" : forecast_ci.iloc[:, 0].values.astype(int),
    "Upper_95" : forecast_ci.iloc[:, 1].values.astype(int),
})
arima_df.to_csv("arima_forecast_2013_2019.csv", index=False)
print("\n  ✔  arima_forecast_2013_2019.csv percapita_crime_rates.csv saved")

# ── Plot 10 : ARIMA Forecast ──────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 11))

# Top panel — full forecast with CI band
ax = axes[0]

# Historical actual
ax.plot(ts.index, ts.values,
        color=PALETTE["primary"], linewidth=2.5,
        marker="o", markersize=7,
        markerfacecolor="white", markeredgewidth=2,
        markeredgecolor=PALETTE["primary"],
        label="Actual  (2001–2012)", zorder=5)

# In-sample fitted
ax.plot(fitted.index[1:], fitted.values[1:],
        color=PALETTE["teal"], linewidth=1.8,
        linestyle="--", alpha=0.8,
        label="ARIMA fitted  (in-sample)", zorder=4)

# Forecast line
ax.plot(forecast_years, forecast_mean.values,
        color=PALETTE["accent"], linewidth=2.2,
        marker="D", markersize=6,
        markerfacecolor=PALETTE["accent"],
        label="ARIMA forecast  (2013–2019)", zorder=5)

# 95% confidence band
ax.fill_between(forecast_years,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color=PALETTE["accent"], alpha=0.12,
                label="95% confidence interval")

# Divider between history and forecast
ax.axvline(2012.5, color=PALETTE["muted"],
           linewidth=1, linestyle=":",
           label="Forecast boundary")

# Annotate each forecast point
for yr, val in zip(forecast_years, forecast_mean.values):
    ax.annotate(f'{int(val):,}',
                xy=(yr, val), xytext=(0, 12),
                textcoords="offset points",
                ha="center", fontsize=7.5,
                color=PALETTE["accent"])

ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
ax.set_xticks(list(ts.index) + forecast_years)
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=8.5, framealpha=0.88, loc="upper left")
style_axis(ax,
           title="ARIMA(1,1,1) Forecast  |  National IPC Crimes  2001 – 2019",
           subtitle="Dashed = in-sample fit  |  Solid red = forecast  |  Shaded = 95% CI",
           xlabel="Year", ylabel="Total IPC Crimes")

# Bottom panel — model comparison: Linear vs Poly vs ARIMA
ax2 = axes[1]

# Actual data
ax2.scatter(ts.index, ts.values,
            color=PALETTE["primary"], s=75, zorder=6,
            label="Actual  (2001–2012)")

# Build comparison series for the forecast window only
lr_fc   = [int(model_lr.predict([[yr]])[0])  for yr in forecast_years]
poly_fc = [int(poly_model.predict([[yr]])[0]) for yr in forecast_years]
arima_fc = forecast_mean.values.tolist()

ax2.plot(forecast_years, lr_fc,
         color=PALETTE["warm"], linewidth=1.8,
         linestyle="--", marker="s", markersize=5,
         label=f"Linear regression  (R²={r2:.3f})")
ax2.plot(forecast_years, poly_fc,
         color=PALETTE["purple"], linewidth=1.8,
         linestyle="-.", marker="^", markersize=5,
         label=f"Polynomial deg-2  (R²={r2_p:.3f})")
ax2.plot(forecast_years, arima_fc,
         color=PALETTE["accent"], linewidth=2.2,
         marker="D", markersize=6,
         label=f"ARIMA(1,1,1)  (R²={r2_a:.3f})")
ax2.fill_between(forecast_years,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color=PALETTE["accent"], alpha=0.10)

ax2.axvline(2012.5, color=PALETTE["muted"],
            linewidth=1, linestyle=":")
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
ax2.set_xticks(list(ts.index) + forecast_years)
ax2.tick_params(axis="x", rotation=45)
ax2.legend(fontsize=8.5, framealpha=0.88, loc="upper left")
style_axis(ax2,
           title="Model Comparison  |  Linear vs Polynomial vs ARIMA  (Forecast 2013–2019)",
           subtitle="All three models shown side-by-side in the forecast window",
           xlabel="Year", ylabel="Projected IPC Crimes")

fig.tight_layout(pad=3.0)
save_fig(fig, "11_arima_forecast")

print("\n  ARIMA model complete.")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  PROJECT COMPLETE")
print("=" * 60)
print(f"""
  Outputs
  ───────────────────────────────────────────────────────────
  Figure 1  —  National crime trend line
  Figure 2  —  Top 10 states bar chart
  Figure 3  —  Crime-type growth 2001→2012
  Figure 4  —  Year-over-year growth rate
  Figure 5  —  State × Year heatmap
  Figure 6  —  Linear + Polynomial forecast
  Figure 7  —  Per-crime-type forecasts
  Figure 8  —  Model accuracy dashboard
  Figure 9  —  Raw vs per-capita comparison
  Figure 10 —  Anomaly detection chart
  Figure 11 —  ARIMA forecast & model comparison

  national_trend.csv
  crime_growth_2001_2012.csv
  crime_type_forecast_2013_2019.csv
  arima_forecast_2013_2019.csv
  percapita_crime_rates.csv
  ───────────────────────────────────────────────────────────
""")
