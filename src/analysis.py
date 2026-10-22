"""Анализ и визуализация: экспорт и импорт важнейших видов продукции Беларуси, 2000–2021."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).parent))
from load import load_all

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "charts"
OUT.mkdir(exist_ok=True)

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
C_EXP, C_IMP = "#1f77b4", "#d62728"

SHORT = {
    "Лекарственные средства, расфасованные для розничной продажи": "Лекарства (розница)",
    "Мясо и пищевые субпродукты домашней птицы": "Мясо птицы",
    "Лесоматериалы продольно-распиленные": "Пиломатериалы",
    "Плиты древесно-стружечные": "ДСП",
    "Холодильники, морозильники и холодильное оборудование": "Холодильники",
    "Части и принадлежности для автомобилей и тракторов": "Автозапчасти",
    "Вычислительные машины для автоматической обработки информации": "Вычислит. машины",
    "Машины и механические устройства специального назначения": "Спецмашины и механизмы",
    "Металлоконструкции из черных металлов": "Металлоконструкции",
    "Отходы, полученные при извлечении соевого масла": "Соевый жмых/шрот",
    "Молоко и сливки несгущенные": "Молоко несгущ.",
    "Молоко и сливки сгущенные и сухие": "Молоко сгущ./сухое",
    "Мебель, включая медицинскую": "Мебель",
    "Провода изолированные, кабели": "Провода и кабели",
    "Сельскохозяйственная техника": "С/х техника",
    "Химические волокна и нити": "Хим. волокна",
    "Двигатели внутреннего сгорания": "Двигатели ВС",
    "Аппаратура связи и части к ней": "Аппаратура связи",
    "Трубы из черных металлов": "Трубы",
    "Масло сливочное": "Сливочное масло",
}
short = lambda p: SHORT.get(p, p)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


df = load_all()
df["short"] = df["product"].map(short)
# приводим легковые автомобили (экспорт в штуках) к тыс. шт. — как в импорте
mask = (df.flow == "Экспорт") & (df["product"] == "Легковые автомобили")
df.loc[mask, "value"] /= 1000
df.loc[mask, "unit"] = "тыс. шт."

W = {f: g.pivot(index="year", columns="short", values="value") for f, g in df.groupby("flow")}
UNIT = {(r.flow, r.short): r.unit for r in df.drop_duplicates(["flow", "short"]).itertuples()}
EXP, IMP = W["Экспорт"], W["Импорт"]

# ---------- 1. Таблица ключевых метрик ----------
def metrics(w, flow):
    rows = []
    for c in w.columns:
        s = w[c]; yoy = s.pct_change()
        rows.append({
            "Поток": flow, "Продукт": c, "Ед.изм.": UNIT[(flow, c)],
            "2000": s[2000], "2021": s[2021], "Рост 2021/2000, раз": s[2021] / s[2000],
            "CAGR 2000–2021, %": ((s[2021] / s[2000]) ** (1 / 21) - 1) * 100,
            "CAGR 2010–2021, %": ((s[2021] / s[2010]) ** (1 / 11) - 1) * 100,
            "Пик, год": s.idxmax(), "Пик, знач.": s.max(),
            "Коэф. вариации": s.std() / s.mean(),
            "Макс. прирост г/г, %": yoy.max() * 100, "Год макс. прироста": yoy.idxmax(),
            "Макс. спад г/г, %": yoy.min() * 100, "Год макс. спада": yoy.idxmin(),
        })
    return pd.DataFrame(rows)

M = pd.concat([metrics(EXP, "Экспорт"), metrics(IMP, "Импорт")]).round(2)

# ---------- 2. Small multiples: динамика всех позиций ----------
def small_multiples(w, flow, color, fname, ncols=4):
    n = w.shape[1]; nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.4 * nrows), sharex=True)
    for ax, c in zip(axs.ravel(), w.columns):
        ax.plot(w.index, w[c], color=color, lw=1.8)
        ax.fill_between(w.index, w[c], alpha=.12, color=color)
        ax.axvspan(2008.5, 2009.5, color="grey", alpha=.15); ax.axvspan(2014.5, 2015.5, color="grey", alpha=.15)
        ax.set_title(f"{c}, {UNIT[(flow, c)]}", fontsize=8.5)
        ax.set_ylim(bottom=0)
    for ax in axs.ravel()[n:]: ax.axis("off")
    fig.suptitle(f"{flow} важнейших видов продукции, 2000–2021 (серым — 2009 и 2015)", y=1.0, fontsize=12)
    fig.tight_layout(); save(fig, fname)

small_multiples(EXP, "Экспорт", C_EXP, "01_export_small_multiples.png")
small_multiples(IMP, "Импорт", C_IMP, "02_import_small_multiples.png")

# ---------- 3. CAGR ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 6))
for ax, (flow, color) in zip(axs, [("Экспорт", C_EXP), ("Импорт", C_IMP)]):
    d = M[M["Поток"] == flow].sort_values("CAGR 2000–2021, %")
    ax.barh(d["Продукт"], d["CAGR 2000–2021, %"], color=color, alpha=.85)
    ax.axvline(0, color="k", lw=.8)
    for y, v in enumerate(d["CAGR 2000–2021, %"]): ax.text(v + (.3 if v >= 0 else -.3), y, f"{v:.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_title(f"{flow}: среднегодовой темп роста (физ. объём), 2000–2021, %")
fig.tight_layout(); save(fig, "03_cagr.png")

# ---------- 4. Тепловая карта индекса (среднее 2010–2012 = 100) ----------
def idx_heat(w, flow, ax):
    base = w.loc[2010:2012].mean()
    h = (w / base * 100).T
    h = h.loc[h.loc[:, 2019:2021].mean(axis=1).sort_values().index]
    im = ax.imshow(np.log10(h.clip(lower=1)), aspect="auto", cmap="RdYlBu_r", vmin=np.log10(10), vmax=np.log10(1000)) if False else \
         ax.imshow(np.log2(h.clip(lower=3)/100), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(len(h.columns))); ax.set_xticklabels(h.columns, rotation=90)
    ax.set_yticks(range(len(h.index))); ax.set_yticklabels(h.index)
    ax.set_title(f"{flow}: индекс объёма (среднее 2010–2012 = 100); красный — выше, синий — ниже", fontsize=10)
    ax.grid(False)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.iloc[i, j]
            if j % 3 == 0 or j == h.shape[1] - 1:
                ax.text(j, i, f"{v:.0f}" if v < 10000 else ">1e4", ha="center", va="center", fontsize=6)
fig, axs = plt.subplots(2, 1, figsize=(13, 13), gridspec_kw={"height_ratios": [19, 13]})
idx_heat(EXP, "Экспорт", axs[0]); idx_heat(IMP, "Импорт", axs[1])
fig.tight_layout(); save(fig, "04_index_heatmap.png")

# ---------- 5. Структурные сдвиги: рост молочки/мяса и древесных товаров ----------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for c in ["Говядина", "Мясо птицы", "Сыры и творог", "Молоко несгущ.", "Молоко сгущ./сухое", "Сливочное масло"]:
    axs[0].plot(EXP.index, EXP[c], label=c, lw=2)
axs[0].set_title("Агропродовольственный экспорт, тыс. т"); axs[0].legend(fontsize=7)
axs[1].plot(EXP.index, EXP["Пиломатериалы"], color="#8c564b", lw=2, label="Пиломатериалы, тыс. м³")
axs[1].plot(EXP.index, EXP["ДСП"], color="#e377c2", lw=2, label="ДСП, тыс. м³")
axs[1].set_title("Экспорт древесных материалов"); axs[1].legend(fontsize=8)
for c, col in [("С/х техника", "#2ca02c"), ("Провода и кабели", "#ff7f0e"), ("Мебель", "#9467bd")]:
    s = EXP[c]; axs[2].plot(EXP.index, s / s.loc[2000:2008].mean() * 100, label=f"{c}", lw=2, color=col)
axs[2].set_title("Машиностроение/промтовары: индекс (среднее 2000–2008 = 100)"); axs[2].legend(fontsize=8)
fig.tight_layout(); save(fig, "05_export_structural_shifts.png")

# ---------- 6. Автомобили: экспорт vs импорт ----------
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(IMP.index, IMP["Легковые автомобили"], color=C_IMP, lw=2.2, label="Импорт легковых, тыс. шт.")
ax.plot(EXP.index, EXP["Легковые автомобили"], color=C_EXP, lw=2.2, label="Экспорт легковых, тыс. шт.")
ax.axvline(2011, color="grey", ls="--", lw=.8)
ax.annotate("2011: пик импорта (284 тыс.)\nи экспорта (113 тыс.)", (2011, 284), (2013.3, 255), arrowprops=dict(arrowstyle="->"), fontsize=8)
ax.annotate("2012: импорт −71%", (2012, 83.7), (2013.5, 160), arrowprops=dict(arrowstyle="->"), fontsize=8)
ax.set_title("Легковые автомобили: ввоз и вывоз"); ax.legend(); fig.tight_layout(); save(fig, "06_cars_export_vs_import.png")

# ---------- 7. Товары, присутствующие и в экспорте, и в импорте ----------
common = [("Черные металлы", "тыс. т"), ("Автозапчасти", "тыс. т"), ("Лекарства (розница)", "тонн"), ("Легковые автомобили", "тыс. шт.")]
fig, axs = plt.subplots(2, 4, figsize=(16, 6.5), gridspec_kw={"height_ratios": [3, 2]})
corr_rows = []
for k, (c, u) in enumerate(common):
    ax = axs[0, k]
    ax.plot(EXP.index, EXP[c], color=C_EXP, lw=2, label="Экспорт"); ax.plot(IMP.index, IMP[c], color=C_IMP, lw=2, label="Импорт")
    ax.set_title(f"{c}, {u}"); ax.legend(fontsize=7)
    net = EXP[c] - IMP[c]
    axs[1, k].bar(net.index, net, color=np.where(net >= 0, C_EXP, C_IMP), alpha=.8)
    axs[1, k].set_title("Сальдо (экспорт − импорт), физ. объём", fontsize=8)
    corr_rows.append({"Продукт": c, "corr уровней": EXP[c].corr(IMP[c]),
                      "corr г/г": EXP[c].pct_change().corr(IMP[c].pct_change()),
                      "Экспорт/импорт 2021, %": EXP[c][2021] / IMP[c][2021] * 100,
                      "Экспорт/импорт 2000, %": EXP[c][2000] / IMP[c][2000] * 100})
fig.tight_layout(); save(fig, "07_common_products_balance.png")

# ---------- 8. Корреляция годовых приростов ----------
ALL = pd.concat({"Э": EXP, "И": IMP}, axis=1)
ALL.columns = [f"{a}: {b}" for a, b in ALL.columns]
G = ALL.pct_change().iloc[1:]
G = G.clip(-1, 1.5)  # ограничиваем выбросы (напр. +1600% в молоке 2001), чтобы не искажали корреляцию
CM = G.corr(method="spearman")
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(CM, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(len(CM))); ax.set_xticklabels(CM.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(CM))); ax.set_yticklabels(CM.index, fontsize=7)
ax.axhline(len(EXP.columns) - .5, color="k", lw=1); ax.axvline(len(EXP.columns) - .5, color="k", lw=1)
fig.colorbar(im, shrink=.7); ax.set_title("Ранговая корреляция (Спирмен) годовых приростов, 2001–2021\n(Э — экспорт, И — импорт)")
fig.tight_layout(); save(fig, "08_growth_correlation.png")
pairs = CM.where(np.triu(np.ones(CM.shape), 1).astype(bool)).stack().dropna().sort_values()

# ---------- 9. Кризисные годы ----------
yoy = ALL.pct_change() * 100
fig, axs = plt.subplots(1, 3, figsize=(16, 7), sharey=False)
for ax, yr in zip(axs, [2009, 2015, 2020]):
    s = yoy.loc[yr].clip(-100, 150).sort_values()
    ax.barh(s.index, s, color=np.where(s.index.str.startswith("Э"), C_EXP, C_IMP))
    ax.axvline(0, color="k", lw=.8); ax.set_title(f"{yr}: изменение к предыдущему году, %"); ax.tick_params(axis="y", labelsize=6.5)
fig.suptitle("Реакция на внешние шоки (синий — экспорт, красный — импорт; ось обрезана на −100/+150%)", y=1.0)
fig.tight_layout(); save(fig, "09_shock_years.png")

# ---------- 10. Аномалии: z-score по приросту ----------
Z = (G - G.mean()) / G.std()
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(Z.T.clip(-4, 4), aspect="auto", cmap="PuOr_r", vmin=-4, vmax=4); ax.grid(False)
ax.set_xticks(range(len(Z))); ax.set_xticklabels(Z.index, rotation=90)
ax.set_yticks(range(Z.shape[1])); ax.set_yticklabels(Z.columns, fontsize=7); fig.colorbar(im, shrink=.6)
ax.set_title("Аномальность годовых приростов (z-score; обрезано ±4)"); fig.tight_layout(); save(fig, "10_anomaly_zscore.png")
