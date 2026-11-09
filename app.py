import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import linprog

st.set_page_config(page_title="Optimizador de Concreto Premezclado", page_icon="🏗️", layout="wide")

LOTE_BASE = 2400.0  # kg, lote de referencia del caso CEMEX

DEFAULT = pd.DataFrame(
    {
        "Insumo": ["Cemento CPC 30R", "Ceniza volante", "Arena de mina", "Grava triturada ¾"],
        "Costo ($/kg)": [3.60, 1.90, 0.50, 0.35],
        "Resistencia f'c": [0.65, 0.40, 0.05, 0.08],
        "Pasta ligante (%)": [35.0, 25.0, 5.0, 2.0],
        "Absorción (%)": [2.0, 4.0, 6.0, 1.0],
        "Disponib. máx (kg, 0 = sin límite)": [0.0, 300.0, 0.0, 0.0],
    }
)


def resolver(df, lote, fc_min, pasta_pct, abs_pct):
    c = df["Costo ($/kg)"].to_numpy(float)
    fc = df["Resistencia f'c"].to_numpy(float)
    pasta = df["Pasta ligante (%)"].to_numpy(float) / 100
    absor = df["Absorción (%)"].to_numpy(float) / 100
    disp = df["Disponib. máx (kg, 0 = sin límite)"].to_numpy(float)

    # Forma A_ub x <= b_ub
    A_ub = np.array([-fc, -pasta, absor])
    b_ub = np.array([-fc_min, -pasta_pct / 100 * lote, abs_pct / 100 * lote])
    A_eq = np.ones((1, 4))
    b_eq = np.array([lote])
    bounds = [(0, (u if u > 0 else None)) for u in disp]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res


st.title("🏗️ Optimizador de costos – Concreto premezclado f'c = 250 kg/cm²")
st.caption("Programación lineal (SciPy/HiGHS) · Caso práctico CEMEX · Se recalcula en tiempo real")

# ---------------- Entradas ----------------
with st.sidebar:
    st.header("⚙️ Parámetros del lote")
    lote = st.number_input("Tamaño del lote (kg)", 500.0, 10000.0, LOTE_BASE, 50.0)
    escalar = st.checkbox("Escalar f'c mínima proporcionalmente al lote", value=True,
                          help="Si está activo, la f'c mínima se multiplica por lote/2400.")
    fc_base = st.slider("f'c mínima acumulada (base 2,400 kg)", 100.0, 400.0, 250.0, 5.0)
    pasta_pct = st.slider("Pasta ligante mínima (% del lote)", 5.0, 30.0, 15.0, 0.5)
    abs_pct = st.slider("Absorción máxima de agua (% del lote)", 1.0, 8.0, 3.5, 0.1)
    if st.button("↺ Restablecer valores"):
        st.session_state.pop("tabla", None)
        st.rerun()

fc_min = fc_base * (lote / LOTE_BASE if escalar else 1.0)

st.subheader("📋 Datos de insumos (editables)")
df = st.data_editor(
    DEFAULT,
    key="tabla",
    use_container_width=True,
    hide_index=True,
    disabled=["Insumo"],
    column_config={
        "Costo ($/kg)": st.column_config.NumberColumn(min_value=0.0, step=0.05, format="%.2f"),
        "Resistencia f'c": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="%.2f"),
        "Pasta ligante (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5),
        "Absorción (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1),
        "Disponib. máx (kg, 0 = sin límite)": st.column_config.NumberColumn(min_value=0.0, step=10.0),
    },
)

# ---------------- Optimización ----------------
if df.isnull().any().any():
    st.error("⚠️ Hay celdas vacías en la tabla. Complétalas para calcular.")
    st.stop()

res = resolver(df, lote, fc_min, pasta_pct, abs_pct)

if res.status == 0:
    x = res.x
    costo = res.fun
    st.success("✅ Solución óptima encontrada")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Costo total del lote", f"${costo:,.2f}")
    m2.metric("Costo por kg", f"${costo / lote:,.4f}")
    m3.metric("Costo por tonelada", f"${costo / lote * 1000:,.2f}")
    m4.metric("Masa total", f"{x.sum():,.1f} kg")

    tabla = pd.DataFrame(
        {
            "Insumo": df["Insumo"],
            "Cantidad (kg)": x.round(2),
            "% del lote": (x / lote * 100).round(2),
            "Costo ($)": (x * df["Costo ($/kg)"]).round(2),
        }
    )
    tabla["% del costo"] = (tabla["Costo ($)"] / costo * 100).round(2)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Composición óptima")
        st.dataframe(tabla, hide_index=True, use_container_width=True)
    with c2:
        tab_a, tab_b = st.tabs(["Proporción en masa", "Proporción en costo"])
        with tab_a:
            fig = px.pie(tabla, names="Insumo", values="Cantidad (kg)", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with tab_b:
            fig = px.bar(tabla, x="Insumo", y="Costo ($)", text="Costo ($)")
            st.plotly_chart(fig, use_container_width=True)

    # Verificación de restricciones
    st.subheader("🔎 Verificación de restricciones")
    fc_val = float(x @ df["Resistencia f'c"].to_numpy(float))
    pasta_val = float(x @ (df["Pasta ligante (%)"].to_numpy(float) / 100))
    abs_val = float(x @ (df["Absorción (%)"].to_numpy(float) / 100))
    ver = pd.DataFrame(
        {
            "Restricción": ["f'c acumulada", "Pasta ligante (kg)", "Absorción de agua (kg)"],
            "Requisito": [f"≥ {fc_min:,.1f}", f"≥ {pasta_pct / 100 * lote:,.1f}", f"≤ {abs_pct / 100 * lote:,.1f}"],
            "Valor logrado": [round(fc_val, 2), round(pasta_val, 2), round(abs_val, 2)],
            "Holgura": [
                round(fc_val - fc_min, 2),
                round(pasta_val - pasta_pct / 100 * lote, 2),
                round(abs_pct / 100 * lote - abs_val, 2),
            ],
        }
    )
    ver["Estado"] = np.where(ver["Holgura"].abs() < 1e-4, "🔴 Activa (limitante)", "🟢 Con holgura")
    st.dataframe(ver, hide_index=True, use_container_width=True)

    # Precios sombra
    with st.expander("📈 Análisis de sensibilidad (precios sombra)"):
        m = res.ineqlin.marginals  # d(costo)/d(b_ub)
        sombra = pd.DataFrame(
            {
                "Restricción": ["f'c mínima", "Pasta ligante mínima", "Absorción máxima"],
                "Efecto en costo ($) por +1 unidad del requisito": [
                    round(-m[0], 4), round(-m[1], 4), round(m[2], 4)
                ],
            }
        )
        st.dataframe(sombra, hide_index=True, use_container_width=True)
        st.caption(
            "Ej.: si el valor de f'c es 0.6, subir en 1 el requisito de f'c aumentaría "
            "el costo mínimo en ≈ $0.60. Un valor 0 indica restricción no limitante."
        )

elif res.status == 2:
    st.error("❌ **Modelo infactible**: no existe ninguna mezcla que cumpla todas las restricciones a la vez.")
    st.markdown(
        "**Posibles causas / qué ajustar:**\n"
        "- Resistencia f'c mínima demasiado alta para los insumos disponibles.\n"
        "- Pasta ligante mínima muy alta o disponibilidad de ceniza/cemento insuficiente.\n"
        "- Absorción máxima demasiado baja (el cemento y la grava absorben poco; la arena mucho).\n"
        "- Combinación de f'c alta + absorción baja que obliga a usar más cemento del que cabe en el lote."
    )
    with st.expander("Detalle técnico"):
        st.code(res.message)
elif res.status == 3:
    st.warning("⚠️ El problema es no acotado. Revisa los datos de entrada.")
else:
    st.warning(f"⚠️ El solver no pudo resolver el modelo: {res.message}")

with st.expander("📐 Modelo matemático"):
    st.latex(r"\min Z = c_1x_1 + c_2x_2 + c_3x_3 + c_4x_4")
    st.latex(r"\sum f_i x_i \ge f'c_{min},\quad \sum p_i x_i \ge P_{min},\quad \sum a_i x_i \le A_{max}")
    st.latex(r"x_2 \le D_{ceniza},\quad \sum x_i = \text{Lote},\quad x_i \ge 0")
