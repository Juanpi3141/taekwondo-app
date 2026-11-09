import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Física del taekwondo", layout="wide")
st.title("Física del taekwondo: simulación de patadas")

TECNICAS = {
    "Dollyeo chagi (circular)": 180,
    "Ap chagi (frontal)": 90,
    "Yeop chagi (lateral)": 120,
    "Dwit chagi (trasera)": 180,
}

# ---------- Configuración ----------
st.sidebar.header("Configuración")
tecnica = st.sidebar.selectbox("Técnica", list(TECNICAS.keys()))
sweep_deg = st.sidebar.number_input(
    "Ángulo de barrido (°) — supuesto", 30.0, 270.0, float(TECNICAS[tecnica]), 5.0
)
L = st.sidebar.number_input("Longitud de la pierna L (m)", 0.5, 1.2, 0.9, 0.05)
m_pierna = st.sidebar.number_input("Masa de la pierna (kg)", 1.0, 20.0, 8.0, 0.5)
m_eff = st.sidebar.number_input("Masa efectiva en el impacto (kg)", 0.5, 10.0, 3.0, 0.5)
w_max = st.sidebar.number_input("Velocidad angular máxima (rad/s)", 1.0, 40.0, 15.0, 0.5)
t_acc = st.sidebar.number_input("Tiempo de aceleración (s)", 0.05, 1.0, 0.4, 0.05)
dt_contacto = st.sidebar.number_input("Duración del contacto (s)", 0.005, 0.1, 0.02, 0.005)

st.sidebar.header("Visualización")
v_min, v_max = st.sidebar.slider("Rango válido de velocidad (m/s)", -100.0, 100.0, (-40.0, 40.0))
n_frames = st.sidebar.slider("Cuadros de la animación", 20, 200, 90)
retardo = st.sidebar.slider("Retardo entre cuadros (s)", 0.01, 0.2, 0.04, 0.01)


def velocidad_valida(v, nombre):
    if v_min <= v <= v_max:
        return True
    st.error(f"{nombre} = {v:.2f} m/s está fuera del rango [{v_min}, {v_max}] m/s.")
    return False


# ---------- Modelo ----------
def simular(L, m, w_max, t_acc, sweep_rad):
    """Pierna como barra de masa m que gira alrededor de la cadera.
    Fase 1: aceleración angular constante hasta w_max.
    Fase 2: velocidad angular constante hasta el impacto."""
    alpha = w_max / t_acc
    th_acc = 0.5 * w_max * t_acc
    if th_acc > sweep_rad:
        return None
    t_cruce = (sweep_rad - th_acc) / w_max
    t_imp = t_acc + t_cruce
    t = np.linspace(0, t_imp, 300)
    omega = np.minimum(alpha * t, w_max)
    theta = np.where(t <= t_acc, 0.5 * alpha * t**2, th_acc + w_max * (t - t_acc))
    I = m * L**2 / 3                  # momento de inercia de una barra respecto a un extremo
    v_tip = omega * L                 # v = ωL
    ke = 0.5 * I * omega**2           # K = ½ I ω²
    p = m * omega * L / 2             # p = m·v_cm, con v_cm = ωL/2
    return dict(t=t, omega=omega, theta=theta, v_tip=v_tip, ke=ke, p=p, t_imp=t_imp)


def figura(res, idx, L, sweep):
    t, th = res["t"], res["theta"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Trayectoria de la punta
    ang = np.linspace(0, sweep, 200)
    ax1.plot(L * np.cos(ang), L * np.sin(ang), "k--", lw=0.8, alpha=0.5)
    ax1.plot(L * np.cos(th[: idx + 1]), L * np.sin(th[: idx + 1]),
             color="tab:red", alpha=0.35, lw=4)
    tx, ty = L * np.cos(th[idx]), L * np.sin(th[idx])
    ax1.plot([0, tx], [0, ty], "k-", lw=3)
    ax1.plot(tx, ty, "o", color="tab:red", ms=10)
    ax1.plot(0, 0, "ko", ms=6)
    ax1.plot(L * np.cos(sweep), L * np.sin(sweep), "X", color="tab:gray", ms=12)
    # Velocidad tangencial de la punta (verde)
    s = 0.3 * L
    ax1.annotate("", xy=(tx - s * np.sin(th[idx]), ty + s * np.cos(th[idx])), xytext=(tx, ty),
                 arrowprops=dict(arrowstyle="->", color="tab:green", lw=2))
    lim = 1.3 * L
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)
    ax1.set_title("Punta del pie (X = objetivo)")

    # Velocidad y energía cinética
    ax2.plot(t[: idx + 1], res["v_tip"][: idx + 1], color="tab:green", label="v punta")
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(0, res["v_tip"].max() * 1.1)
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("v (m/s)")
    ax3 = ax2.twinx()
    ax3.plot(t[: idx + 1], res["ke"][: idx + 1], color="tab:purple", ls="--", label="K")
    ax3.set_ylim(0, res["ke"].max() * 1.1)
    ax3.set_ylabel("Energía cinética (J)")
    ax2.axvline(t[idx], color="gray", lw=0.8)
    ax2.grid(alpha=0.3)
    ax2.set_title("Velocidad de la punta y energía")
    fig.tight_layout()
    return fig


# ---------- Pestañas ----------
tab_sim, tab_teoria = st.tabs(["Simulación", "Teoría"])

with tab_teoria:
    st.subheader("Física detrás de la patada")
    st.markdown("La pierna se modela como una barra uniforme que gira alrededor de la cadera.")
    st.latex(r"I = \tfrac{1}{3} m L^2 \qquad \text{(momento de inercia respecto al extremo)}")
    st.latex(r"v_{punta} = \omega L \qquad v_{cm} = \tfrac{1}{2}\omega L")
    st.latex(r"K = \tfrac{1}{2} I \omega^2 \qquad p = m\,v_{cm}")
    st.latex(r"\alpha = \frac{\omega_{max}}{t_{acc}} \qquad \theta_{acc} = \tfrac{1}{2}\omega_{max}t_{acc}")
    st.latex(r"\Delta p = m_{eff}\,v_{punta} \qquad F_{prom} = \frac{\Delta p}{\Delta t_{contacto}}")
    st.markdown(
        "Un golpe rápido no depende solo de la fuerza muscular: depende de la velocidad "
        "de la punta, que crece linealmente con la longitud de la pierna y la velocidad "
        "angular. Como la energía cinética crece con el cuadrado de ω, pequeños aumentos "
        "de velocidad angular producen grandes aumentos de energía. Un contacto más corto "
        "aumenta la fuerza media sobre el objetivo."
    )

with tab_sim:
    st.subheader(tecnica)
    sweep = np.radians(sweep_deg)
    res = simular(L, m_pierna, w_max, t_acc, sweep)

    if res is None:
        st.error("El tiempo de aceleración es demasiado largo para el ángulo de barrido. "
                 "Reduce t_acc, sube la velocidad angular o aumenta el ángulo.")
    else:
        v_imp = res["v_tip"][-1]
        p_imp = m_eff * v_imp
        F_imp = p_imp / dt_contacto
        E_imp = 0.5 * m_eff * v_imp**2

        if velocidad_valida(v_imp, "Velocidad de la punta al impacto"):
            st.latex(r"\omega = \frac{\omega_{max}}{t_{acc}}\,t, \qquad v = \omega L")

            met = st.columns(4)
            ph_v = met[0].empty()
            ph_k = met[1].empty()
            ph_p = met[2].empty()
            ph_th = met[3].empty()
            placeholder = st.empty()

            def mostrar_metricas(idx):
                ph_v.metric("v punta (m/s)", f"{res['v_tip'][idx]:.2f}")
                ph_k.metric("K (J)", f"{res['ke'][idx]:.2f}")
                ph_p.metric("p cm (kg·m/s)", f"{res['p'][idx]:.2f}")
                ph_th.metric("Ángulo (°)", f"{np.degrees(res['theta'][idx]):.1f}")

            if st.button("▶ Animar patada"):
                for k in range(n_frames):
                    idx = int(k / (n_frames - 1) * (len(res["t"]) - 1))
                    mostrar_metricas(idx)
                    fig = figura(res, idx, L, sweep)
                    placeholder.pyplot(fig)
                    plt.close(fig)
                    time.sleep(retardo)
                st.success("Impacto simulado.")
            else:
                mostrar_metricas(len(res["t"]) - 1)
                fig = figura(res, len(res["t"]) - 1, L, sweep)
                placeholder.pyplot(fig)
                plt.close(fig)

            st.subheader("Resultados en el impacto")
            r1, r2, r3 = st.columns(3)
            r1.metric("Cantidad de movimiento transferida (kg·m/s)", f"{p_imp:.2f}")
            r2.metric("Fuerza media en el contacto (N)", f"{F_imp:.0f}")
            r3.metric("Energía cinética en el impacto (J)", f"{E_imp:.1f}")
