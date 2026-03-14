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

def calc_Rs(presion, presion_burbuja, Rs_inicial, API, gamma_g, temperatura):
    """
    Rs [scf/stb] — Relación de gas disuelto.
    Si presion >= presion_burbuja: Rs = Rs_inicial (undersaturated, gas permanece en solución).
    Si presion < presion_burbuja:  Correlación de Standing.
    # Standing (1947)
    """
    if presion >= presion_burbuja:
        return Rs_inicial
    rs = gamma_g * ((presion / 18.2 + 1.4) * (10 ** (0.0125 * API - 0.00091 * temperatura))) ** 1.205
    return max(float(rs), 0.0)

def calc_Bo(Rs, gamma_g, gamma_o, temperatura):
    """
    Bo [rb/stb] — Factor volumétrico del petróleo.
    # Standing (1947)
    """
    F = Rs * (gamma_g / gamma_o) ** 0.5 + 1.25 * temperatura
    bo = 0.9759 + 0.000120 * (F ** 1.2)
    return float(np.clip(bo, 1.04, 1.81))

def calc_Bg(presion, temperatura, factor_z=0.9):
    """
    Bg [rb/scf] — Factor volumétrico del gas.
    Bg = 0.00504 * z * T_R / presion  (T_R en Rankine)
    # Craft & Hawkins (1959)
    """
    temperatura_rankine = temperatura + 459.67
    if presion <= 0:
        return 1e-6
    bg = 0.00504 * factor_z * temperatura_rankine / presion
    return max(float(bg), 1e-6)

def calc_Rs_burbuja(presion_burbuja, API, gamma_g, temperatura):
    """Rs en condiciones de burbuja (presion = presion_burbuja)."""
    return gamma_g * ((presion_burbuja / 18.2 + 1.4) * (10 ** (0.0125 * API - 0.00091 * temperatura))) ** 1.205

# =============================================================================
# GENERACIÓN DE PARÁMETROS ALEATORIOS POR POZO
# =============================================================================

def generar_params(pozo_id):
    presion_inicial     = np.random.uniform(2500, 5000)
    presion_burbuja     = np.random.uniform(0.5 * presion_inicial, 0.9 * presion_inicial)
    API                 = np.random.uniform(25, 45)
    gamma_g             = np.random.uniform(0.65, 0.85)
    gamma_o             = 141.5 / (API + 131.5)          # gravedad específica del aceite
    temperatura         = np.random.uniform(150, 250)     # °F
    porosidad           = np.random.uniform(0.10, 0.28)
    permeabilidad       = np.random.uniform(5, 200)
    espesor             = np.random.uniform(10, 80)       # m
    area                = np.random.uniform(500_000, 5_000_000)  # m²
    caudal_inicial      = np.random.uniform(200, 2000)    # caudal inicial de aceite [bbl/d]
    declinacion_inicial = np.random.uniform(0.0005, 0.003)
    exponente_arps      = np.random.uniform(0.3, 1.0)
    VRR                 = np.random.uniform(0.8, 1.2)
    water_cut_rate      = np.random.uniform(0.0005, 0.002)  # tasa de crecimiento [1/día]
    water_cut_max       = np.random.uniform(0.4, 0.7)
    duracion_dias       = int(np.random.uniform(3 * 365, 15 * 365))

    Rs_burbuja = float(calc_Rs_burbuja(presion_burbuja, API, gamma_g, temperatura))
    Bo_inicial = calc_Bo(Rs_burbuja, gamma_g, gamma_o, temperatura)

    # Volumen poroso en barriles: Vp [bbl] = A[m²] × h[m] × φ × 6.28981 (conv m³→bbl)
    volumen_poroso_bbl    = area * espesor * porosidad * 6.28981
    compresibilidad_total = np.random.uniform(10e-6, 25e-6)  # [1/psi]
    Bw       = np.random.uniform(0.99, 1.05)     # factor volumétrico del agua [rb/stb]
    factor_z = np.random.uniform(0.85, 0.95)     # factor de compresibilidad del gas

    return {
        "pozo_id": pozo_id,
        "presion_inicial": presion_inicial, "presion_burbuja": presion_burbuja,
        "API": API, "gamma_g": gamma_g, "gamma_o": gamma_o, "temperatura": temperatura,
        "porosidad": porosidad, "permeabilidad": permeabilidad,
        "espesor": espesor, "area": area,
        "caudal_inicial": caudal_inicial, "declinacion_inicial": declinacion_inicial,
        "exponente_arps": exponente_arps, "VRR": VRR,
        "water_cut_rate": water_cut_rate, "water_cut_max": water_cut_max,
        "duracion_dias": duracion_dias, "Rs_burbuja": Rs_burbuja, "Bo_inicial": Bo_inicial,
        "volumen_poroso_bbl": volumen_poroso_bbl,
        "compresibilidad_total": compresibilidad_total, "Bw": Bw, "factor_z": factor_z,
    }

# =============================================================================
# CONSTRUCCIÓN DEL VECTOR DE TIEMPOS
# =============================================================================

def build_timesteps(duracion_dias):
    tiempos = []
    dia = 0
    while dia <= duracion_dias:
        tiempos.append(dia)
        if dia < 30:
            dia += 1
        elif dia < 365:
            dia += 3
        else:
            dia += 7
    return np.array(tiempos)

# =============================================================================
# SIMULACIÓN DE UN POZO
# =============================================================================

def simular_pozo(params):
    pozo_id             = params["pozo_id"]
    presion_inicial     = params["presion_inicial"]
    presion_burbuja     = params["presion_burbuja"]
    API                 = params["API"]
    gamma_g             = params["gamma_g"]
    gamma_o             = params["gamma_o"]
    temperatura         = params["temperatura"]
    Rs_burbuja          = params["Rs_burbuja"]
    porosidad           = params["porosidad"]
    permeabilidad       = params["permeabilidad"]
    espesor             = params["espesor"]
    area                = params["area"]
    caudal_inicial      = params["caudal_inicial"]
    declinacion_inicial = params["declinacion_inicial"]
    exponente_arps      = params["exponente_arps"]
    VRR                 = params["VRR"]
    water_cut_rate      = params["water_cut_rate"]
    water_cut_max       = params["water_cut_max"]
    volumen_poroso_bbl  = params["volumen_poroso_bbl"]
    compresibilidad_total = params["compresibilidad_total"]
    Bw                  = params["Bw"]
    factor_z            = params["factor_z"]

    timesteps = build_timesteps(params["duracion_dias"])

    # Definir 2 shut-ins aleatorios sin solapamiento (inicio, duración)
    duracion = params["duracion_dias"]
    shutin_upper = max(duracion - 50, 201)
    shutin_duracion_1 = int(np.random.randint(15, 46))
    shutin_inicio_1 = int(np.random.randint(200, shutin_upper))
    shutin_fin_1 = min(shutin_inicio_1 + shutin_duracion_1, duracion)
    # Segundo shut-in empieza después de que termina el primero
    shutin_inicio_2_min = shutin_fin_1 + 30  # al menos 30 días de separación
    if shutin_inicio_2_min < shutin_upper:
        shutin_inicio_2 = int(np.random.randint(shutin_inicio_2_min, shutin_upper))
    else:
        shutin_inicio_2 = shutin_fin_1 + 30
    shutin_duracion_2 = int(np.random.randint(15, 46))
    shutin_fin_2 = min(shutin_inicio_2 + shutin_duracion_2, duracion)
    shutin_intervalos = [
        (shutin_inicio_1, shutin_fin_1),
        (shutin_inicio_2, shutin_fin_2),
    ]

    def en_shutin(tiempo):
        for inicio, fin in shutin_intervalos:
            if inicio <= tiempo <= fin:
                return True
        return False

    # Estado inicial
    presion                  = presion_inicial
    acum_petroleo            = 0.0
    acum_gas                 = 0.0
    acum_agua                = 0.0
    acum_inyeccion           = 0.0
    voidage_acum_petroleo_rb = 0.0  # voidage acumulado de petróleo+gas libre [rb]
    tiempo_efectivo          = 0.0  # tiempo efectivo de producción (excluye shut-ins)

    rows = []

    for i, tiempo in enumerate(timesteps):
        delta_t = float(timesteps[i] - timesteps[i-1]) if i > 0 else 0.0

        shutin = en_shutin(tiempo)

        # --- Caudal de aceite (Arps hiperbólico con tiempo efectivo) ---
        if shutin:
            caudal_petroleo = 0.0
        else:
            caudal_petroleo = float(
                caudal_inicial / (1 + exponente_arps * declinacion_inicial * tiempo_efectivo) ** (1 / exponente_arps)
            )
            caudal_petroleo = max(caudal_petroleo, 0.0)

        # --- PVT a presión actual ---
        Rs = calc_Rs(presion, presion_burbuja, Rs_burbuja, API, gamma_g, temperatura)
        Bo = calc_Bo(Rs, gamma_g, gamma_o, temperatura)
        Bg = calc_Bg(presion, temperatura, factor_z)

        # --- Caudal de gas (disuelto + libre) ---
        if shutin:
            caudal_gas_mscfd = 0.0
            voidage_petroleo_rb = 0.0
            gas_libre_scfd = 0.0
        else:
            gas_libre_scfd = max(caudal_petroleo * (Rs_burbuja - Rs), 0.0)        # [scf/d]
            caudal_gas_total_scfd = caudal_petroleo * Rs + gas_libre_scfd          # [scf/d]
            caudal_gas_mscfd = caudal_gas_total_scfd / 1000.0                      # [Mscf/d]
            # Voidage: Bo ya incluye gas en solución, solo sumar gas libre
            voidage_petroleo_rb = caudal_petroleo * Bo + gas_libre_scfd * Bg

        # --- Producción de agua simplificada (water cut creciente, parametrizado) ---
        if shutin:
            caudal_agua = 0.0
            water_cut = 0.0
        else:
            water_cut = min(water_cut_max, water_cut_rate * tiempo_efectivo)
            caudal_agua = caudal_petroleo * water_cut / max(1.0 - water_cut, 0.01)

        # --- Inyección de agua (empieza en día 180) ---
        if shutin or tiempo < 180:
            caudal_inyeccion = 0.0
        else:
            voidage_total = voidage_petroleo_rb + caudal_agua * Bw  # [rb/d]
            caudal_inyeccion = max(VRR * voidage_total / Bw, 0.0)  # [bbl/d]

        # --- Actualizar acumuladas ---
        if delta_t > 0:
            acum_petroleo  += caudal_petroleo    * delta_t
            acum_gas       += caudal_gas_mscfd * 1000 * delta_t   # convertir a scf
            acum_agua      += caudal_agua        * delta_t
            acum_inyeccion += caudal_inyeccion   * delta_t
            voidage_acum_petroleo_rb += voidage_petroleo_rb * delta_t

        # --- Actualizar presión (MBE diferencial) ---
        if delta_t > 0:
            if shutin:
                # Recuperación exponencial hacia presión de equilibrio depletada
                presion_equilibrio = presion_inicial - (
                    voidage_acum_petroleo_rb + acum_agua * Bw - acum_inyeccion * Bw
                ) / (volumen_poroso_bbl * compresibilidad_total)
                presion_equilibrio = max(presion_equilibrio, presion_burbuja * 0.3)
                presion = presion + (presion_equilibrio - presion) * (1 - np.exp(-0.03 * delta_t))
            else:
                voidage_neto_rb = voidage_petroleo_rb + caudal_agua * Bw - caudal_inyeccion * Bw
                delta_presion = -(voidage_neto_rb * delta_t) / (volumen_poroso_bbl * compresibilidad_total)
                presion += delta_presion

        presion = float(np.clip(presion, presion_burbuja * 0.3, presion_inicial))

        # Acumular tiempo efectivo de producción
        if not shutin and delta_t > 0:
            tiempo_efectivo += delta_t

        rows.append({
            "pozo_id":                   pozo_id,
            "tiempo_dias":               int(tiempo),
            "Porosidad":                 round(porosidad, 4),
            "Permeabilidad_mD":          round(permeabilidad, 3),
            "Espesor_Neto_m":            round(espesor, 2),
            "Area_m2":                   round(area, 1),
            "Presion_Burbuja_psi":       round(presion_burbuja, 2),
            "Bo_rb_stb":                 round(Bo, 5),
            "Bg_rb_scf":                 round(Bg, 8),
            "Rs_scf_stb":                round(Rs, 2),
            "Caudal_Prod_Petroleo_bbl":  round(caudal_petroleo, 2),
            "Caudal_Prod_Gas_Mscf":      round(caudal_gas_mscfd, 4),
            "Caudal_Iny_Agua_bbl":       round(caudal_inyeccion, 2),
            "Water_Cut":                 round(water_cut, 4),
            "Prod_Acumulada_Petroleo":   round(acum_petroleo, 1),
            "Prod_Acumulada_Gas":        round(acum_gas, 1),
            "Prod_Acumulada_Agua":       round(acum_agua, 1),
            "Iny_Acumulada_Agua":        round(acum_inyeccion, 1),
            "Presion_Reservorio_psi":    round(presion, 2),
        })

    return pd.DataFrame(rows)

# =============================================================================
# APLICAR RUIDO GAUSSIANO (σ = 1.5%)
# =============================================================================

def aplicar_ruido(df, presiones_iniciales):
    presion_max = df["pozo_id"].map(presiones_iniciales)

    # Identificar shut-in antes de aplicar ruido (caudal_petroleo == 0 indica cierre)
    shutin_mask = df["Caudal_Prod_Petroleo_bbl"] == 0

    # Ruido en caudales
    for col in ["Caudal_Prod_Petroleo_bbl", "Caudal_Prod_Gas_Mscf", "Caudal_Iny_Agua_bbl"]:
        ruido = np.random.normal(0, 0.015, len(df))
        df[col] = (df[col] * (1 + ruido)).clip(lower=0)

    # Ruido en presión solo en períodos productivos (shut-in mantiene presion sin ruido)
    ruido_presion = np.random.normal(0, 0.015, len(df))
    presion_con_ruido = (df["Presion_Reservorio_psi"] * (1 + ruido_presion)).clip(lower=0)
    df.loc[~shutin_mask, "Presion_Reservorio_psi"] = presion_con_ruido[~shutin_mask]
    df["Presion_Reservorio_psi"] = df["Presion_Reservorio_psi"].clip(upper=presion_max)

    # Recalcular acumuladas desde caudales con ruido para mantener consistencia
    for pozo_id, grupo in df.groupby("pozo_id"):
        idx = grupo.index
        delta_t = grupo["tiempo_dias"].diff().fillna(0).values
        df.loc[idx, "Prod_Acumulada_Petroleo"] = np.round(
            (grupo["Caudal_Prod_Petroleo_bbl"].values * delta_t).cumsum(), 1)
        df.loc[idx, "Prod_Acumulada_Gas"] = np.round(
            (grupo["Caudal_Prod_Gas_Mscf"].values * 1000 * delta_t).cumsum(), 1)
        df.loc[idx, "Iny_Acumulada_Agua"] = np.round(
            (grupo["Caudal_Iny_Agua_bbl"].values * delta_t).cumsum(), 1)

    # Redondeo
    df["Presion_Reservorio_psi"]   = df["Presion_Reservorio_psi"].round(2)
    df["Caudal_Prod_Petroleo_bbl"] = df["Caudal_Prod_Petroleo_bbl"].round(2)
    df["Caudal_Prod_Gas_Mscf"]     = df["Caudal_Prod_Gas_Mscf"].round(4)
    df["Caudal_Iny_Agua_bbl"]      = df["Caudal_Iny_Agua_bbl"].round(2)
    return df

# =============================================================================
# VALIDACIÓN
# =============================================================================

def validar(df, presiones_iniciales):
    errores = []

    # Presión no supera la inicial por pozo
    for pozo_id, grupo in df.groupby("pozo_id"):
        presion_inicial_ref = presiones_iniciales[pozo_id]
        if grupo["Presion_Reservorio_psi"].max() > presion_inicial_ref * 1.001:
            errores.append(f"{pozo_id}: presion supera presion_inicial")

    # Acumuladas no decrecientes
    for pozo_id, grupo in df.groupby("pozo_id"):
        for col in ["Prod_Acumulada_Petroleo", "Prod_Acumulada_Gas",
                    "Prod_Acumulada_Agua", "Iny_Acumulada_Agua"]:
            if (grupo[col].diff().dropna() < -1e-3).any():
                errores.append(f"{pozo_id}: {col} decrece")

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
        for error in errores:
            print(f"   • {error}")
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
    presiones_iniciales = {}   # {pozo_id: presion_inicial} guardado antes de cualquier ruido

    for i in range(1, N_POZOS + 1):
        pozo_id = f"W{i:03d}"
        params = generar_params(pozo_id)
        presiones_iniciales[pozo_id] = params["presion_inicial"]
        df_pozo = simular_pozo(params)
        dfs.append(df_pozo)
        print(f"  {pozo_id}: {len(df_pozo)} filas | Pr_i={params['presion_inicial']:.0f} psi | Pb={params['presion_burbuja']:.0f} psi | {params['duracion_dias']} días")

    df = pd.concat(dfs, ignore_index=True)
    df = aplicar_ruido(df, presiones_iniciales)

    # Validar
    print("\nEjecutando validaciones físicas...")
    validar(df, presiones_iniciales)

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

    # Rango de presion por pozo (primeros 10)
    print("\n── Rango de Presion_Reservorio_psi por pozo (primeros 10) ──────")
    rango_presion = df.groupby("pozo_id")["Presion_Reservorio_psi"].agg(["min", "max", "mean"]).round(1)
    print(rango_presion.head(10).to_string())
