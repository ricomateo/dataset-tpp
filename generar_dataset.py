"""
generar_dataset.py
==================
Genera un dataset sintético físicamente consistente de presión de reservorio
para entrenamiento de modelos de Machine Learning.

Física implementada:
  - Correlaciones PVT de Standing (1947)
  - Balance de Materiales diferencial (MBE)
  - Declinación de Arps hiperbólico
  - Inyección de agua con VRR variable
  - Shut-ins con recuperación de presión exponencial

Uso:
  pip install numpy pandas
  python generar_dataset.py
  → Produce: dataset_reservorio.csv
"""

import numpy as np
import pandas as pd
import sys
import os

np.random.seed(42)
output_file = "dataset_reservorio.csv"

# =============================================================================
# CORRELACIONES PVT (Standing, 1947)
# =============================================================================

def calc_Rs(Pr, Pb, Rs_i, API, gamma_g, T_F):
    """
    Rs [scf/stb] — Relación de gas disuelto.
    Si Pr >= Pb: Rs = Rs_i (undersaturated, gas permanece en solución).
    Si Pr < Pb:  Correlación de Standing.
    # Standing (1947)
    """
    if Pr >= Pb:
        return Rs_i
    rs = gamma_g * ((Pr / 18.2 + 1.4) ** 1.205) * (10 ** (0.0125 * API - 0.00091 * T_F))
    return max(float(rs), 0.0)

def calc_Bo(Rs, gamma_g, gamma_o, T_F):
    """
    Bo [rb/stb] — Factor volumétrico del petróleo.
    # Standing (1947)
    """
    F = Rs * (gamma_g / gamma_o) ** 0.5 + 1.25 * T_F
    bo = 0.972 + 0.000147 * (F ** 1.175)
    return float(np.clip(bo, 1.05, 1.80))

def calc_Bg(Pr, T_F, z=0.9):
    """
    Bg [rb/scf] — Factor volumétrico del gas.
    Bg = 0.00504 * z * T_R / Pr  (T_R en Rankine)
    # Craft & Hawkins (1959)
    """
    T_R = T_F + 459.67
    bg = 0.00504 * z * T_R / Pr
    return max(float(bg), 1e-6)

def calc_Rs_burbuja(Pb, API, gamma_g, T_F):
    """Rs en condiciones de burbuja (Pr = Pb)."""
    return gamma_g * ((Pb / 18.2 + 1.4) ** 1.205) * (10 ** (0.0125 * API - 0.00091 * T_F))

# =============================================================================
# GENERACIÓN DE PARÁMETROS ALEATORIOS POR POZO
# =============================================================================

def generar_params(pozo_id):
    Pr_i    = np.random.uniform(2500, 5000)
    Pb      = np.random.uniform(0.5 * Pr_i, 0.9 * Pr_i)
    API     = np.random.uniform(25, 45)
    gamma_g = np.random.uniform(0.65, 0.85)
    gamma_o = 141.5 / (API + 131.5)          # gravedad específica del aceite
    T_F     = np.random.uniform(150, 250)     # temperatura en °F
    phi     = np.random.uniform(0.10, 0.28)
    k       = np.random.uniform(5, 200)
    h       = np.random.uniform(10, 80)       # m
    A       = np.random.uniform(500_000, 5_000_000)  # m²
    qi      = np.random.uniform(200, 2000)    # caudal inicial de aceite [bbl/d]
    Di      = np.random.uniform(0.0005, 0.003)
    b       = np.random.uniform(0.3, 1.0)
    VRR     = np.random.uniform(0.8, 1.2)
    duracion_dias = int(np.random.uniform(3 * 365, 15 * 365))

    Rs_i = float(calc_Rs_burbuja(Pb, API, gamma_g, T_F))
    Bo_i = calc_Bo(Rs_i, gamma_g, gamma_o, T_F)

    # Volumen poroso en barriles: Vp [bbl] = A[m²] × h[m] × φ × 6.28981 (conv m³→bbl)
    Vp_bbl = A * h * phi * 6.28981
    ct = 15e-6  # compresibilidad total [1/psi]

    return {
        "pozo_id": pozo_id, "Pr_i": Pr_i, "Pb": Pb, "API": API,
        "gamma_g": gamma_g, "gamma_o": gamma_o, "T_F": T_F,
        "phi": phi, "k": k, "h": h, "A": A,
        "qi": qi, "Di": Di, "b": b, "VRR": VRR,
        "duracion_dias": duracion_dias, "Rs_i": Rs_i, "Bo_i": Bo_i,
        "Vp_bbl": Vp_bbl, "ct": ct,
    }

# =============================================================================
# CONSTRUCCIÓN DEL VECTOR DE TIEMPOS
# =============================================================================

def build_timesteps(duracion_dias):
    t = []
    day = 0
    while day <= duracion_dias:
        t.append(day)
        if day < 30:
            day += 1
        elif day < 365:
            day += 3
        else:
            day += 7
    return np.array(t)

# =============================================================================
# SIMULACIÓN DE UN POZO
# =============================================================================

def simular_pozo(p):
    pid     = p["pozo_id"]
    Pr_i    = p["Pr_i"];  Pb   = p["Pb"]
    API     = p["API"];   gamma_g = p["gamma_g"];  gamma_o = p["gamma_o"]
    T_F     = p["T_F"];   Rs_i = p["Rs_i"]
    phi     = p["phi"];   k    = p["k"];  h = p["h"];  A = p["A"]
    qi      = p["qi"];    Di   = p["Di"]; b = p["b"];  VRR = p["VRR"]
    Vp_bbl  = p["Vp_bbl"];  ct = p["ct"]
    Bw      = 1.0   # rb/stb

    timesteps = build_timesteps(p["duracion_dias"])

    # Definir 2 shut-ins aleatorios (inicio, duración)
    shutin_starts = sorted(np.random.randint(200, p["duracion_dias"] - 50, size=2))
    shutin_durations = np.random.randint(15, 46, size=2)
    shutin_intervals = [(s, s + d) for s, d in zip(shutin_starts, shutin_durations)]

    def en_shutin(t):
        for s, e in shutin_intervals:
            if s <= t <= e:
                return True
        return False

    # Estado inicial
    Pr   = Pr_i
    Np   = 0.0
    Gp   = 0.0
    Wp   = 0.0
    Winj = 0.0

    rows = []

    for i, t in enumerate(timesteps):
        dt = float(timesteps[i] - timesteps[i-1]) if i > 0 else 0.0

        shutin = en_shutin(t)

        # --- Caudal de aceite (Arps hiperbólico) ---
        if shutin:
            qo = 0.0
        else:
            qo = float(qi / (1 + b * Di * t) ** (1 / b))
            qo = max(qo, 0.0)

        # --- PVT a presión actual ---
        Rs = calc_Rs(Pr, Pb, Rs_i, API, gamma_g, T_F)
        Bo = calc_Bo(Rs, gamma_g, gamma_o, T_F)
        Bg = calc_Bg(Pr, T_F)

        # --- Caudal de gas ---
        Qg_mscfd = (qo * Rs / 1000.0) if not shutin else 0.0   # [Mscf/d]

        # --- Inyección de agua (empieza en día 180) ---
        if shutin or t < 180:
            qwinj = 0.0
        else:
            voidage = qo * Bo + Qg_mscfd * 1000 * Bg   # [rb/d]
            qwinj = max(VRR * voidage / Bw, 0.0)        # [bbl/d]

        # --- Actualizar acumuladas ---
        if dt > 0:
            Np   += qo     * dt
            Gp   += Qg_mscfd * 1000 * dt   # convertir a scf
            Winj += qwinj  * dt

        # --- Actualizar presión (MBE diferencial) ---
        if dt > 0:
            if shutin:
                # Recuperación exponencial durante shut-in
                Pr += (Pr_i - Pr) * 0.03 * dt
            else:
                voidage_rb = qo * Bo + Qg_mscfd * 1000 * Bg - qwinj * Bw
                dPr = -(voidage_rb * dt) / (Vp_bbl * ct)
                Pr += dPr

        Pr = float(np.clip(Pr, Pb * 0.3, Pr_i))

        rows.append({
            "pozo_id":                   pid,
            "tiempo_dias":               int(t),
            "Porosidad":                 round(phi, 4),
            "Permeabilidad_mD":          round(k, 3),
            "Espesor_Neto_m":            round(h, 2),
            "Area_m2":                   round(A, 1),
            "Presion_Burbuja_psi":       round(Pb, 2),
            "Bo_rb_stb":                 round(Bo, 5),
            "Bg_rb_scf":                 round(Bg, 8),
            "Rs_scf_stb":                round(Rs, 2),
            "Caudal_Prod_Petroleo_bbl":  round(qo, 2),
            "Caudal_Prod_Gas_Mscf":      round(Qg_mscfd, 4),
            "Caudal_Iny_Agua_bbl":       round(qwinj, 2),
            "Prod_Acumulada_Petroleo":   round(Np, 1),
            "Prod_Acumulada_Gas":        round(Gp, 1),
            "Prod_Acumulada_Agua":       round(Wp, 1),
            "Iny_Acumulada_Agua":        round(Winj, 1),
            "Presion_Reservorio_psi":    round(Pr, 2),
        })

    return pd.DataFrame(rows)

# =============================================================================
# APLICAR RUIDO GAUSSIANO (σ = 1.5%)
# =============================================================================

def aplicar_ruido(df, pr_iniciales):
    cols_ruido = [
        "Presion_Reservorio_psi", "Caudal_Prod_Petroleo_bbl",
        "Caudal_Prod_Gas_Mscf", "Caudal_Iny_Agua_bbl"
    ]
    # Mapear Pr_inicial pre-ruido por pozo (diccionario guardado antes de simular)
    pr_max = df["pozo_id"].map(pr_iniciales)

    for col in cols_ruido:
        ruido = np.random.normal(0, 0.015, len(df))
        df[col] = (df[col] * (1 + ruido)).clip(lower=0)

    # Clipear Pr usando el Pr_inicial original (antes de cualquier ruido)
    df["Presion_Reservorio_psi"] = df["Presion_Reservorio_psi"].clip(upper=pr_max)
    df["Presion_Reservorio_psi"]   = df["Presion_Reservorio_psi"].round(2)
    df["Caudal_Prod_Petroleo_bbl"] = df["Caudal_Prod_Petroleo_bbl"].round(2)
    df["Caudal_Prod_Gas_Mscf"]     = df["Caudal_Prod_Gas_Mscf"].round(4)
    df["Caudal_Iny_Agua_bbl"]      = df["Caudal_Iny_Agua_bbl"].round(2)
    return df

# =============================================================================
# VALIDACIÓN
# =============================================================================

def validar(df, pr_iniciales):
    errores = []

    # Presión no supera la inicial por pozo
    for pid, grp in df.groupby("pozo_id"):
        Pr_i_ref = pr_iniciales[pid]
        if grp["Presion_Reservorio_psi"].max() > Pr_i_ref * 1.001:
            errores.append(f"{pid}: Pr supera Pr_inicial")

    # Acumuladas no decrecientes
    for pid, grp in df.groupby("pozo_id"):
        for col in ["Prod_Acumulada_Petroleo", "Prod_Acumulada_Gas",
                    "Prod_Acumulada_Agua", "Iny_Acumulada_Agua"]:
            if (grp[col].diff().dropna() < -1e-3).any():
                errores.append(f"{pid}: {col} decrece")

    # Bo en rango físico
    if not df["Bo_rb_stb"].between(1.04, 1.81).all():
        errores.append("Bo fuera de rango [1.04, 1.81]")

    # Rs no negativo
    if (df["Rs_scf_stb"] < 0).any():
        errores.append("Rs negativo detectado")

    # Sin NaN
    if df.isnull().any().any():
        errores.append("Se detectaron valores NaN")

    # Mínimo de filas
    if len(df) < 150_000:
        errores.append(f"Dataset tiene solo {len(df)} filas (mínimo 150.000)")

    if errores:
        print("\n❌ ERRORES DE VALIDACIÓN:")
        for e in errores:
            print(f"   • {e}")
        sys.exit(1)
    else:
        print("✅ Todas las validaciones físicas pasaron correctamente.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    N_POZOS = 450
    print(f"Simulando {N_POZOS} pozos (~200.000 filas esperadas)...")

    dfs = []
    params_log = []
    pr_iniciales = {}   # {pozo_id: Pr_inicial} guardado antes de cualquier ruido

    for i in range(1, N_POZOS + 1):
        pid = f"W{i:03d}"
        p = generar_params(pid)
        pr_iniciales[pid] = p["Pr_i"]   # guardar ANTES del ruido
        df_pozo = simular_pozo(p)
        dfs.append(df_pozo)
        params_log.append({
            "pozo_id":           pid,
            "Pr_inicial_psi":    round(p["Pr_i"], 1),
            "Pb_psi":            round(p["Pb"], 1),
            "Porosidad":         round(p["phi"], 3),
            "Permeabilidad_mD":  round(p["k"], 1),
            "Espesor_Neto_m":    round(p["h"], 1),
            "Area_m2":           round(p["A"], 0),
            "qi_bbl_d":          round(p["qi"], 1),
            "Di_1_d":            round(p["Di"], 5),
            "b_Arps":            round(p["b"], 3),
            "VRR":               round(p["VRR"], 3),
            "duracion_dias":     p["duracion_dias"],
        })
        print(f"  {pid}: {len(df_pozo)} filas | Pr_i={p['Pr_i']:.0f} psi | Pb={p['Pb']:.0f} psi | {p['duracion_dias']} días")

    df = pd.concat(dfs, ignore_index=True)
    df = aplicar_ruido(df, pr_iniciales)

    # Validar
    print("\nEjecutando validaciones físicas...")
    validar(df, pr_iniciales)

    # Guardar CSV
    df.to_csv(output_file, index=False)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n📁 Dataset guardado: {output_file}")
    print(f"   Filas totales : {len(df):,}")
    print(f"   Columnas      : {len(df.columns)}")
    print(f"   Tamaño aprox. : {file_size_mb:.1f} MB")

    # Estadísticas descriptivas
    print("\n── Estadísticas descriptivas (columnas numéricas) ───────────────")
    desc = df.describe().loc[["mean", "std", "min", "max"]].T
    desc.columns = ["media", "std", "min", "max"]
    print(desc.round(3).to_string())

    # Rango de Pr por pozo (primeros 10)
    print("\n── Rango de Presion_Reservorio_psi por pozo (primeros 10) ──────")
    rango_pr = df.groupby("pozo_id")["Presion_Reservorio_psi"].agg(["min", "max", "mean"]).round(1)
    print(rango_pr.head(10).to_string())
