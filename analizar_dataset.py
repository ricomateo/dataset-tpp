"""
analizar_dataset.py
===================
Procesa el dataset generado por generar_dataset.py y extrae
un resumen con los valores más relevantes para revisión profesional.
Genera visualizaciones en la carpeta figuras/.

Uso:
  pip install numpy pandas matplotlib
  python analizar_dataset.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = "dataset_reservorio.csv"
FIGURAS_DIR = Path("figuras")

def guardar_fig(nombre):
    path = FIGURAS_DIR / f"{nombre}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Guardada: {path}")


def generar_visualizaciones(df):
    FIGURAS_DIR.mkdir(exist_ok=True)

    n_pozos = df["pozo_id"].nunique()
    pozos = df["pozo_id"].unique()

    # --- 1. Presión vs tiempo (pozos representativos) ---
    np.random.seed(0)
    muestra = np.random.choice(pozos, size=min(6, n_pozos), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
    fig.suptitle("Presión de Reservorio vs Tiempo — Pozos Representativos", fontsize=13)
    for ax, pid in zip(axes.flat, muestra):
        pozo = df[df["pozo_id"] == pid]
        ax.plot(pozo["tiempo_dias"] / 365, pozo["Presion_Reservorio_psi"],
                linewidth=0.8, color="steelblue")
        ax.set_title(pid, fontsize=10)
        ax.set_xlabel("Tiempo [años]")
        ax.set_ylabel("Presión [psi]")
        ax.grid(True, alpha=0.3)
    guardar_fig("01_presion_vs_tiempo")

    # --- 2. Caudal de petróleo vs tiempo (mismos pozos) ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
    fig.suptitle("Caudal de Petróleo vs Tiempo — Pozos Representativos", fontsize=13)
    for ax, pid in zip(axes.flat, muestra):
        pozo = df[df["pozo_id"] == pid]
        ax.plot(pozo["tiempo_dias"] / 365, pozo["Caudal_Prod_Petroleo_bbl"],
                linewidth=0.8, color="darkgreen")
        ax.set_title(pid, fontsize=10)
        ax.set_xlabel("Tiempo [años]")
        ax.set_ylabel("Qo [bbl/d]")
        ax.grid(True, alpha=0.3)
    guardar_fig("02_caudal_petroleo_vs_tiempo")

    # --- 3. Water cut vs tiempo (mismos pozos, excluyendo shut-ins) ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
    fig.suptitle("Water Cut vs Tiempo — Pozos Representativos", fontsize=13)
    for ax, pid in zip(axes.flat, muestra):
        pozo = df[df["pozo_id"] == pid]
        produciendo = pozo[pozo["Caudal_Prod_Petroleo_bbl"] > 0]
        ax.plot(produciendo["tiempo_dias"] / 365, produciendo["Water_Cut"],
                linewidth=0.8, color="darkorange")
        ax.set_title(pid, fontsize=10)
        ax.set_xlabel("Tiempo [años]")
        ax.set_ylabel("Water Cut")
        ax.set_ylim(-0.05, 1.0)
        ax.grid(True, alpha=0.3)
    guardar_fig("03_water_cut_vs_tiempo")

    # --- 4. Distribución de presiones iniciales y finales ---
    pr_por_pozo = df.groupby("pozo_id")["Presion_Reservorio_psi"].agg(["first", "last"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Distribución de Presión Inicial y Final por Pozo", fontsize=13)
    ax1.hist(pr_por_pozo["first"], bins=30, alpha=0.75, color="steelblue")
    ax1.set_xlabel("Presión [psi]")
    ax1.set_ylabel("Cantidad de pozos")
    ax1.set_title("Pr inicial")
    ax1.grid(True, alpha=0.3)
    ax2.hist(pr_por_pozo["last"], bins=30, alpha=0.75, color="salmon")
    ax2.set_xlabel("Presión [psi]")
    ax2.set_title("Pr final")
    ax2.grid(True, alpha=0.3)
    guardar_fig("04_distribucion_presiones")

    # --- 5. Bo vs Presión ---
    sample_idx = df.sample(n=min(5000, len(df)), random_state=42).index
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df.loc[sample_idx, "Presion_Reservorio_psi"],
               df.loc[sample_idx, "Bo_rb_stb"],
               s=3, alpha=0.4, color="teal")
    ax.set_xlabel("Presión de Reservorio [psi]")
    ax.set_ylabel("Bo [rb/stb]")
    ax.set_title("Factor Volumétrico del Petróleo (Bo) vs Presión")
    ax.grid(True, alpha=0.3)
    guardar_fig("05_Bo_vs_presion")

    # --- 6. Rs vs Presión ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df.loc[sample_idx, "Presion_Reservorio_psi"],
               df.loc[sample_idx, "Rs_scf_stb"],
               s=3, alpha=0.4, color="darkorchid")
    ax.set_xlabel("Presión de Reservorio [psi]")
    ax.set_ylabel("Rs [scf/stb]")
    ax.set_title("Relación Gas Disuelto (Rs) vs Presión")
    ax.grid(True, alpha=0.3)
    guardar_fig("06_Rs_vs_presion")

    # --- 7. Producción acumulada normalizada vs caída de presión ---
    ultimo = df.groupby("pozo_id").last()
    primero = df.groupby("pozo_id").first()
    caida_pr = primero["Presion_Reservorio_psi"] - ultimo["Presion_Reservorio_psi"]
    vol_poroso = (primero["Area_m2"] * primero["Espesor_Neto_m"]
                  * primero["Porosidad"] * 6.28981)  # bbl
    np_normalizado = ultimo["Prod_Acumulada_Petroleo"] / vol_poroso
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(np_normalizado, caida_pr,
               s=12, alpha=0.5, color="firebrick")
    ax.set_xlabel("Np / Vp [fracción del volumen poroso]")
    ax.set_ylabel("Caída de presión [psi]")
    ax.set_title("Producción Acumulada Normalizada vs Caída de Presión")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.3)
    guardar_fig("07_Np_vs_caida_presion")

    # --- 8. Histograma de propiedades estáticas ---
    props = [
        ("Porosidad", "Porosidad [frac]", "cornflowerblue"),
        ("Permeabilidad_mD", "Permeabilidad [mD]", "goldenrod"),
        ("Espesor_Neto_m", "Espesor Neto [m]", "mediumseagreen"),
        ("Presion_Burbuja_psi", "Presión de Burbuja [psi]", "indianred"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Distribución de Propiedades Estáticas del Reservorio", fontsize=13)
    props_estaticas = df.groupby("pozo_id").first()
    for ax, (col, label, color) in zip(axes.flat, props):
        ax.hist(props_estaticas[col], bins=30, alpha=0.75, color=color)
        ax.set_xlabel(label)
        ax.set_ylabel("Cantidad de pozos")
        ax.grid(True, alpha=0.3)
    guardar_fig("08_propiedades_estaticas")

    # --- 9. Presión + Caudal + Inyección en un mismo gráfico (1 pozo ejemplo) ---
    pid_ejemplo = muestra[0]
    pozo = df[df["pozo_id"] == pid_ejemplo]
    t_anos = pozo["tiempo_dias"] / 365

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Resumen de Producción — {pid_ejemplo}", fontsize=13)

    color_pr = "steelblue"
    ax1.plot(t_anos, pozo["Presion_Reservorio_psi"], color=color_pr, linewidth=1, label="Presión")
    ax1.set_xlabel("Tiempo [años]")
    ax1.set_ylabel("Presión [psi]", color=color_pr)
    ax1.tick_params(axis="y", labelcolor=color_pr)

    ax2 = ax1.twinx()
    ax2.plot(t_anos, pozo["Caudal_Prod_Petroleo_bbl"], color="darkgreen", linewidth=0.8,
             alpha=0.8, label="Qo")
    ax2.plot(t_anos, pozo["Caudal_Iny_Agua_bbl"], color="darkorange", linewidth=0.8,
             alpha=0.8, label="Qw iny")
    ax2.set_ylabel("Caudal [bbl/d]")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    guardar_fig("09_resumen_pozo_ejemplo")

    print(f"\n  Total: 9 figuras guardadas en {FIGURAS_DIR}/")


def main():
    df = pd.read_csv(INPUT_FILE)
    df.sort_values(["pozo_id", "tiempo_dias"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("=" * 70)
    print("RESUMEN DEL DATASET DE PRESIÓN DE RESERVORIO")
    print("=" * 70)

    # --- Dimensiones generales ---
    n_pozos = df["pozo_id"].nunique()
    print(f"\nRegistros totales : {len(df):,}")
    print(f"Pozos simulados   : {n_pozos}")
    print(f"Columnas          : {len(df.columns)}")

    # --- Duración por pozo ---
    duracion = df.groupby("pozo_id")["tiempo_dias"].max()
    print(f"\nDuración de simulación por pozo:")
    print(f"  Mínima : {duracion.min():,} días ({duracion.min() / 365:.1f} años)")
    print(f"  Máxima : {duracion.max():,} días ({duracion.max() / 365:.1f} años)")
    print(f"  Media  : {duracion.mean():,.0f} días ({duracion.mean() / 365:.1f} años)")

    # --- Rangos de propiedades estáticas (un valor por pozo) ---
    props_estaticas = df.groupby("pozo_id").first()
    print("\n" + "-" * 70)
    print("PROPIEDADES ESTÁTICAS DEL RESERVORIO (rango entre pozos)")
    print("-" * 70)
    for col, unidad in [
        ("Porosidad", "frac"),
        ("Permeabilidad_mD", "mD"),
        ("Espesor_Neto_m", "m"),
        ("Area_m2", "m²"),
        ("Presion_Burbuja_psi", "psi"),
    ]:
        vals = props_estaticas[col]
        print(f"  {col:30s} [{unidad:>5s}]  min={vals.min():.2f}  max={vals.max():.2f}  media={vals.mean():.2f}")

    # --- Presión de reservorio ---
    print("\n" + "-" * 70)
    print("PRESIÓN DE RESERVORIO [psi]")
    print("-" * 70)
    pr = df["Presion_Reservorio_psi"]
    print(f"  Global  — min: {pr.min():.1f}  max: {pr.max():.1f}  media: {pr.mean():.1f}")

    pr_por_pozo = df.groupby("pozo_id")["Presion_Reservorio_psi"].agg(["first", "last", "min", "max"])
    pr_por_pozo.columns = ["Pr_inicial", "Pr_final", "Pr_min", "Pr_max"]
    caida = pr_por_pozo["Pr_inicial"] - pr_por_pozo["Pr_final"]
    caida_pct = (caida / pr_por_pozo["Pr_inicial"]) * 100

    print(f"\n  Caída de presión (Pr_inicial - Pr_final):")
    print(f"    Mínima  : {caida.min():.1f} psi ({caida_pct.min():.1f}%)")
    print(f"    Máxima  : {caida.max():.1f} psi ({caida_pct.max():.1f}%)")
    print(f"    Media   : {caida.mean():.1f} psi ({caida_pct.mean():.1f}%)")

    # Pozos que llegan al piso de presión
    pr_burbuja = props_estaticas["Presion_Burbuja_psi"]
    piso = pr_burbuja * 0.3
    pozos_piso = (pr_por_pozo["Pr_min"] <= piso.reindex(pr_por_pozo.index) * 1.01).sum()
    print(f"\n  Pozos que alcanzan el piso de presión (0.3 × Pb): {pozos_piso} / {n_pozos}")

    # --- Producción acumulada ---
    print("\n" + "-" * 70)
    print("PRODUCCIÓN ACUMULADA AL FINAL DE CADA POZO")
    print("-" * 70)
    ultimo = df.groupby("pozo_id").last()
    for col, unidad in [
        ("Prod_Acumulada_Petroleo", "bbl"),
        ("Prod_Acumulada_Gas", "scf"),
        ("Prod_Acumulada_Agua", "bbl"),
        ("Iny_Acumulada_Agua", "bbl"),
    ]:
        vals = ultimo[col]
        print(f"  {col:30s} [{unidad:>5s}]  min={vals.min():>14,.0f}  max={vals.max():>14,.0f}  media={vals.mean():>14,.0f}")

    # --- Caudales ---
    print("\n" + "-" * 70)
    print("CAUDALES DE PRODUCCIÓN (excluyendo shut-ins)")
    print("-" * 70)
    produciendo = df[df["Caudal_Prod_Petroleo_bbl"] > 0]
    for col, unidad in [
        ("Caudal_Prod_Petroleo_bbl", "bbl/d"),
        ("Caudal_Prod_Gas_Mscf", "Mscf/d"),
        ("Caudal_Iny_Agua_bbl", "bbl/d"),
    ]:
        vals = produciendo[col]
        print(f"  {col:30s} [{unidad:>7s}]  min={vals.min():>10.1f}  max={vals.max():>10.1f}  media={vals.mean():>10.1f}")

    # --- Propiedades PVT ---
    print("\n" + "-" * 70)
    print("PROPIEDADES PVT")
    print("-" * 70)
    for col, unidad in [
        ("Bo_rb_stb", "rb/stb"),
        ("Bg_rb_scf", "rb/scf"),
        ("Rs_scf_stb", "scf/stb"),
    ]:
        vals = df[col]
        print(f"  {col:20s} [{unidad:>8s}]  min={vals.min():.5f}  max={vals.max():.5f}  media={vals.mean():.5f}")

    # --- Water cut ---
    print("\n" + "-" * 70)
    print("WATER CUT (al final de cada pozo)")
    print("-" * 70)
    wc_final = ultimo["Water_Cut"]
    print(f"  min  : {wc_final.min():.3f}")
    print(f"  max  : {wc_final.max():.3f}")
    print(f"  media: {wc_final.mean():.3f}")

    # --- Detección de shut-ins ---
    print("\n" + "-" * 70)
    print("SHUT-INS DETECTADOS")
    print("-" * 70)
    shutin_rows = df[df["Caudal_Prod_Petroleo_bbl"] == 0]
    pozos_con_shutin = shutin_rows["pozo_id"].nunique()
    print(f"  Registros en shut-in          : {len(shutin_rows):,}")
    print(f"  Pozos con al menos un shut-in : {pozos_con_shutin} / {n_pozos}")

    # --- Correlaciones clave ---
    print("\n" + "-" * 70)
    print("CORRELACIONES CON PRESIÓN DE RESERVORIO")
    print("-" * 70)
    cols_corr = [
        "Caudal_Prod_Petroleo_bbl", "Caudal_Prod_Gas_Mscf",
        "Caudal_Iny_Agua_bbl", "Water_Cut", "Prod_Acumulada_Petroleo",
        "Bo_rb_stb", "Rs_scf_stb", "tiempo_dias",
    ]
    for col in cols_corr:
        corr = df["Presion_Reservorio_psi"].corr(df[col])
        print(f"  Pr vs {col:30s} : {corr:+.4f}")

    # --- Top 5 pozos con mayor y menor caída de presión ---
    print("\n" + "-" * 70)
    print("TOP 5 POZOS — MAYOR CAÍDA DE PRESIÓN")
    print("-" * 70)
    top_caida = caida.sort_values(ascending=False).head(5)
    for pid, val in top_caida.items():
        pct = caida_pct[pid]
        print(f"  {pid}: {val:,.0f} psi ({pct:.1f}%)")

    print("\n" + "-" * 70)
    print("TOP 5 POZOS — MENOR CAÍDA DE PRESIÓN")
    print("-" * 70)
    bot_caida = caida.sort_values(ascending=True).head(5)
    for pid, val in bot_caida.items():
        pct = caida_pct[pid]
        print(f"  {pid}: {val:,.0f} psi ({pct:.1f}%)")

    # --- Visualizaciones ---
    print("\n" + "-" * 70)
    print("GENERANDO VISUALIZACIONES...")
    print("-" * 70)
    generar_visualizaciones(df)

    print("\n" + "=" * 70)
    print("FIN DEL RESUMEN")
    print("=" * 70)


if __name__ == "__main__":
    main()
