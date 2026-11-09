import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Física del taekwondo", layout="wide")
st.title("Física del taekwondo")

TECNICAS = {
    "Dollyeo chagi (circular)": 180,
    "Ap chagi (frontal)": 90,
    "Yeop chagi (lateral)": 120,
    "Dwit chagi (trasera)": 180,
}

# ---------- Configuración ----------
st.sidebar.header("Técnica y cuerpo")
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

st.sidebar.header("Reproducción")
lento = st.sidebar.slider("Cámara lenta (veces más lento)", 1, 12, 6)
v_min, v_max = st.sidebar.slider("Rango válido de velocidad (m/s)", -100.0, 100.0, (-40.0, 40.0))


def velocidad_valida(v, nombre):
    if v_min <= v <= v_max:
        return True
    st.error(f"{nombre} = {v:.2f} m/s está fuera del rango [{v_min}, {v_max}] m/s.")
    return False


# ---------- Modelo ----------
def simular(L, m, w_max, t_acc, sweep_rad):
    """Pierna como barra uniforme que gira alrededor de la cadera.
    Fase 1: aceleración angular constante hasta w_max.
    Fase 2: velocidad angular constante hasta el impacto."""
    alpha = w_max / t_acc
    th_acc = 0.5 * w_max * t_acc
    if th_acc > sweep_rad:
        return None
    t_imp = t_acc + (sweep_rad - th_acc) / w_max
    t = np.linspace(0, t_imp, 300)
    omega = np.minimum(alpha * t, w_max)
    theta = np.where(t <= t_acc, 0.5 * alpha * t**2, th_acc + w_max * (t - t_acc))
    I = m * L**2 / 3          # barra respecto a un extremo
    return dict(
        t=t,
        theta=theta,
        omega=omega,
        v_tip=omega * L,          # v = ωL
        ke=0.5 * I * omega**2,    # K = ½ I ω²
        p=m * omega * L / 2,      # p = m·v_cm, v_cm = ωL/2
        t_imp=t_imp,
    )


# ---------- Visualización en el navegador (sin Python en cada cuadro) ----------
PLANTILLA = """
<div style="font-family:sans-serif;color:#222">
<svg viewBox="0 0 600 400" width="100%" style="background:#fff;border:1px solid #ddd;border-radius:8px">
  <line x1="0" y1="380" x2="600" y2="380" stroke="#999" stroke-width="2"/>
  <path id="arc" fill="none" stroke="#bbb" stroke-dasharray="6 6"/>
  <path id="rastro" fill="none" stroke="#E24B4A" stroke-opacity="0.35" stroke-width="5"/>
  <line id="stand" stroke="#333" stroke-width="12" stroke-linecap="round"/>
  <line id="torso" stroke="#333" stroke-width="14" stroke-linecap="round"/>
  <line id="arm" stroke="#333" stroke-width="9" stroke-linecap="round"/>
  <circle id="head" r="22" fill="#333"/>
  <line id="kick" stroke="#2b5fa8" stroke-width="13" stroke-linecap="round"/>
  <line id="vec" stroke="#1D9E75" stroke-width="4"/>
  <circle id="vhead" r="5" fill="#1D9E75"/>
  <circle id="foot" r="9" fill="#E24B4A"/>
  <text x="16" y="28" font-size="14" fill="#555">Punta del pie: <tspan id="vtxt">0.00</tspan> m/s</text>
</svg>

<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0;">
  <button id="btn" style="padding:6px 14px;border-radius:6px;border:1px solid #bbb;background:#f5f5f5;cursor:pointer;">▶ Animar patada</button>
  <button id="rst" style="padding:6px 14px;border-radius:6px;border:1px solid #bbb;background:#f5f5f5;cursor:pointer;">↺ Reiniciar</button>
  <span id="tt" style="font-size:14px;">t = 0.000 s</span>
  <span id="estado" style="font-size:14px;color:#666;"></span>
</div>

<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-bottom:10px;">
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">v punta (m/s)</div><div id="mv" style="font-size:20px;font-weight:500;">0.00</div></div>
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">K (J)</div><div id="mk" style="font-size:20px;font-weight:500;">0.00</div></div>
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">p centro de masa (kg·m/s)</div><div id="mp" style="font-size:20px;font-weight:500;">0.00</div></div>
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">Ángulo (°)</div><div id="mt" style="font-size:20px;font-weight:500;">0.0</div></div>
</div>

<svg viewBox="0 0 600 150" width="100%" style="background:#fff;border:1px solid #ddd;border-radius:8px">
  <line x1="10" y1="130" x2="590" y2="130" stroke="#999"/>
  <polyline id="pv" fill="none" stroke="#1D9E75" stroke-width="2"/>
  <polyline id="pk" fill="none" stroke="#7F77DD" stroke-width="2" stroke-dasharray="5 4"/>
  <line id="cur" x1="10" y1="8" x2="10" y2="130" stroke="#888" stroke-dasharray="3 3"/>
  <text x="14" y="20" font-size="12" fill="#1D9E75" id="lv">v máx</text>
  <text x="14" y="36" font-size="12" fill="#7F77DD" id="lk">K máx</text>
</svg>
</div>

<script>
const D = __DATA__;
const S = Math.min(180, 300 / D.L);
const GY = 380;
const hx = 260, hy = GY - D.L * S;
const $ = id => document.getElementById(id);

function pt(th) {
  return [hx + D.L * S * Math.sin(th), hy + D.L * S * Math.cos(th)];
}

// Cuerpo fijo
const shY = hy - 0.6 * S;
$('stand').setAttribute('x1', hx); $('stand').setAttribute('y1', hy);
$('stand').setAttribute('x2', hx - 18); $('stand').setAttribute('y2', GY);
$('torso').setAttribute('x1', hx); $('torso').setAttribute('y1', hy);
$('torso').setAttribute('x2', hx); $('torso').setAttribute('y2', shY);
$('arm').setAttribute('x1', hx); $('arm').setAttribute('y1', shY + 12);
$('arm').setAttribute('x2', hx + 40); $('arm').setAttribute('y2', shY + 60);
$('head').setAttribute('cx', hx); $('head').setAttribute('cy', shY - 26);
$('kick').setAttribute('x1', hx); $('kick').setAttribute('y1', hy);

// Trayectoria (arco) y gráficas
let arc = "";
for (let i = 0; i <= 60; i++) {
  const [x, y] = pt(D.sweep * i / 60);
  arc += (i ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1) + " ";
}
$('arc').setAttribute('d', arc);

const vmax = Math.max(...D.v) * 1.1 || 1;
const kmax = Math.max(...D.k) * 1.1 || 1;
$('lv').textContent = "v máx " + Math.max(...D.v).toFixed(1) + " m/s";
$('lk').textContent = "K máx " + Math.max(...D.k).toFixed(1) + " J";
function poly(arr, max) {
  return D.t.map((t, i) =>
    (10 + 580 * t / D.tend).toFixed(1) + "," + (130 - 120 * arr[i] / max).toFixed(1)
  ).join(" ");
}
$('pv').setAttribute('points', poly(D.v, vmax));
$('pk').setAttribute('points', poly(D.k, kmax));

function frame(i) {
  const th = D.th[i], t = D.t[i];
  const [fx, fy] = pt(th);
  $('kick').setAttribute('x2', fx); $('kick').setAttribute('y2', fy);
  $('foot').setAttribute('cx', fx); $('foot').setAttribute('cy', fy);

  // Vector velocidad: tangente a la trayectoria, longitud proporcional a v
  const len = 4 * D.v[i];
  const vx = fx + Math.cos(th) * len, vy = fy - Math.sin(th) * len;
  $('vec').setAttribute('x1', fx); $('vec').setAttribute('y1', fy);
  $('vec').setAttribute('x2', vx); $('vec').setAttribute('y2', vy);
  $('vhead').setAttribute('cx', vx); $('vhead').setAttribute('cy', vy);

  // Rastro de la punta hasta el instante actual
  let r = "";
  for (let j = 0; j <= i; j++) {
    const [x, y] = pt(D.th[j]);
    r += (j ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1) + " ";
  }
  $('rastro').setAttribute('d', r);

  $('tt').textContent = "t = " + t.toFixed(3) + " s";
  $('vtxt').textContent = D.v[i].toFixed(2);
  $('mv').textContent = D.v[i].toFixed(2);
  $('mk').textContent = D.k[i].toFixed(2);
  $('mp').textContent = D.p[i].toFixed(2);
  $('mt').textContent = (th * 180 / Math.PI).toFixed(1);
  $('cur').setAttribute('x1', 10 + 580 * t / D.tend);
  $('cur').setAttribute('x2', 10 + 580 * t / D.tend);
}

// Reproducción en tiempo real escalada por "cámara lenta"
let elapsed = 0, last = null, playing = false, loopId = 0;
const dur = D.tend * D.slow;

function loop(id, now) {
  if (id !== loopId || !playing) return;
  if (last !== null) elapsed += (now - last) / 1000;
  last = now;
  const f = Math.min(elapsed / dur, 1);
  frame(Math.round(f * (D.t.length - 1)));
  if (f >= 1) {
    playing = false;
    $('btn').textContent = "▶ Repetir";
    $('estado').textContent = "Impacto";
    return;
  }
  requestAnimationFrame(n => loop(id, n));
}

$('btn').onclick = () => {
  if (playing) {
    playing = false; last = null;
    $('btn').textContent = "▶ Continuar";
    $('estado').textContent = "En pausa";
    return;
  }
  if (elapsed >= dur) { elapsed = 0; }
  playing = true; last = null;
  loopId++;
  $('btn').textContent = "⏸ Pausa";
  $('estado').textContent = "Reproduciendo";
  requestAnimationFrame(n => loop(loopId, n));
};

$('rst').onclick = () => {
  playing = false; last = null; elapsed = 0; loopId++;
  $('btn').textContent = "▶ Animar patada";
  $('estado').textContent = "";
  frame(0);
};

frame(0);
</script>
"""

# ---------- Pestañas ----------
tab_sim, tab_teoria = st.tabs(["Simulación", "Teoría"])

with tab_teoria:
    st.subheader("Física detrás de la patada")
    st.markdown("La pierna se modela como una barra uniforme que gira alrededor de la cadera.")
    st.latex(r"I = \tfrac{1}{3} m L^2 \qquad \text{(respecto al extremo)}")
    st.latex(r"v_{punta} = \omega L \qquad v_{cm} = \tfrac{1}{2}\omega L")
    st.latex(r"K = \tfrac{1}{2} I \omega^2 \qquad p = m\,v_{cm}")
    st.latex(r"\alpha = \frac{\omega_{max}}{t_{acc}} \qquad \theta_{acc} = \tfrac{1}{2}\omega_{max}t_{acc}")
    st.latex(r"\Delta p = m_{eff}\,v_{punta} \qquad F_{prom} = \frac{\Delta p}{\Delta t_{contacto}}")
    st.markdown(
        "La velocidad de la punta crece con la longitud de la pierna y la velocidad angular. "
        "Como la energía cinética depende de ω², pequeños aumentos de velocidad angular "
        "producen grandes aumentos de energía. Un contacto más corto aumenta la fuerza media."
    )

with tab_sim:
    st.subheader(tecnica)
    sweep = np.radians(sweep_deg)
    res = simular(L, m_pierna, w_max, t_acc, sweep)

    if res is None:
        st.error("El tiempo de aceleración es demasiado largo para el ángulo de barrido. "
                 "Reduce t_acc, sube la velocidad angular o aumenta el ángulo.")
    else:
        v_imp = float(res["v_tip"][-1])
        if velocidad_valida(v_imp, "Velocidad de la punta al impacto"):
            idx = np.linspace(0, len(res["t"]) - 1, 120).astype(int)
            datos = {
                "L": L,
                "sweep": float(sweep),
                "tend": float(res["t_imp"]),
                "slow": lento,
                "t": res["t"][idx].round(4).tolist(),
                "th": res["theta"][idx].round(4).tolist(),
                "v": res["v_tip"][idx].round(3).tolist(),
                "k": res["ke"][idx].round(3).tolist(),
                "p": res["p"][idx].round(3).tolist(),
            }
            components.html(PLANTILLA.replace("__DATA__", json.dumps(datos)), height=760)

            p_imp = m_eff * v_imp
            F_imp = p_imp / dt_contacto
            E_imp = 0.5 * m_eff * v_imp**2

            st.subheader("Resultados en el impacto")
            r1, r2, r3 = st.columns(3)
            r1.metric("Cantidad de movimiento transferida (kg·m/s)", f"{p_imp:.2f}")
            r2.metric("Fuerza media en el contacto (N)", f"{F_imp:.0f}")
            r3.metric("Energía cinética en el impacto (J)", f"{E_imp:.1f}")
