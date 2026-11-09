import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Física del taekwondo", layout="wide")
st.title("Física del taekwondo")

# Cada técnica tiene su propio plano, sentido y forma de movimiento (valores supuestos)
TECNICAS = {
    "Ap chagi (patada frontal)": dict(
        sweep=90, w_max=7.0, t_acc=0.3, pliegue=75, lean=12, dir=1,
        vista="Vista lateral",
        descr="El pie avanza al frente. La rodilla parte flexionada y se extiende en el impacto.",
    ),
    "Yeop chagi (patada lateral)": dict(
        sweep=120, w_max=8.0, t_acc=0.3, pliegue=60, lean=18, dir=1,
        vista="Vista frontal",
        descr="La pierna sale hacia el costado. El cuerpo se inclina hacia el lado contrario.",
    ),
    "Dollyeo chagi (circular)": dict(
        sweep=150, w_max=12.0, t_acc=0.3, pliegue=100, lean=8, dir=1,
        vista="Vista lateral",
        descr="La rodilla se eleva y la pierna describe un arco hacia el objetivo, con giro de cadera.",
    ),
    "Dwit chagi (trasera)": dict(
        sweep=100, w_max=8.0, t_acc=0.3, pliegue=60, lean=10, dir=-1,
        vista="Vista lateral",
        descr="El cuerpo gira y la pierna se extiende hacia atrás.",
    ),
}

# ---------- Configuración ----------
st.sidebar.header("Técnica y cuerpo")
tecnica = st.sidebar.selectbox("Técnica", list(TECNICAS.keys()))
P = TECNICAS[tecnica]
L = st.sidebar.number_input("Longitud de la pierna L (m)", 0.5, 1.2, 0.9, 0.05)
m_pierna = st.sidebar.number_input("Masa de la pierna (kg)", 1.0, 20.0, 8.0, 0.5)
m_eff = st.sidebar.number_input("Masa efectiva en el impacto (kg)", 0.5, 10.0, 3.0, 0.5)
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
    alpha_t = np.where(t <= t_acc, alpha, 0.0)
    I = m * L**2 / 3                    # barra respecto a un extremo
    return dict(
        t=t, theta=theta, omega=omega, alpha=alpha_t,
        v=omega * L,                    # v = ωL
        k=0.5 * I * omega**2,           # K = ½ I ω²
        p=m * omega * L / 2,            # p = m·v_cm, v_cm = ωL/2
        tau=I * alpha_t,                # τ = I α
        t_imp=t_imp,
    )


def impacto(res, m_eff, dt):
    v = float(res["v"][-1])
    p = m_eff * v
    return dict(v=v, p=p, F=p / dt, K=0.5 * m_eff * v**2)


# ---------- Visualización en el navegador ----------
PLANTILLA = """
<div style="font-family:sans-serif;color:#222">
<svg viewBox="0 0 600 400" width="100%" style="border:1px solid #ddd;border-radius:8px">
  <rect width="600" height="400" fill="#f4f6f8"/>
  <text id="vista" x="590" y="24" font-size="13" fill="#555" text-anchor="end"></text>
  <line x1="0" y1="380" x2="600" y2="380" stroke="#9aa3a8" stroke-width="2"/>
  <path id="arc" fill="none" stroke="#bbb" stroke-dasharray="6 6"/>
  <path id="rastro" fill="none" stroke="#E24B4A" stroke-opacity="0.35" stroke-width="5"/>

  <polyline id="sOut" fill="none" stroke="#5f6b73" stroke-linecap="round" stroke-linejoin="round"/>
  <polyline id="sFill" fill="none" stroke="#ECECEC" stroke-linecap="round" stroke-linejoin="round"/>
  <ellipse id="sShoe" fill="#222"/>

  <g id="upper">
    <polyline id="aOutD" fill="none" stroke="#5f6b73" stroke-linecap="round" stroke-linejoin="round"/>
    <polyline id="aFillD" fill="none" stroke="#ECECEC" stroke-linecap="round" stroke-linejoin="round"/>
    <polyline id="aOutG" fill="none" stroke="#5f6b73" stroke-linecap="round" stroke-linejoin="round"/>
    <polyline id="aFillG" fill="none" stroke="#ECECEC" stroke-linecap="round" stroke-linejoin="round"/>
    <line id="neck" stroke="#E0A77A" stroke-linecap="round"/>
    <polygon id="jacket" fill="#ECECEC" stroke="#5f6b73" stroke-width="2" stroke-linejoin="round"/>
    <line id="belt" stroke="#1a1a1a" stroke-linecap="butt"/>
    <ellipse id="hair" fill="#222"/>
    <circle id="head" fill="#E0A77A" stroke="#8a5a3a" stroke-width="2"/>
    <circle id="eye" fill="#222"/>
  </g>

  <polyline id="kOutT" fill="none" stroke="#5f6b73" stroke-linecap="round" stroke-linejoin="round"/>
  <polyline id="kFillT" fill="none" stroke="#ECECEC" stroke-linecap="round" stroke-linejoin="round"/>
  <polyline id="kOutS" fill="none" stroke="#5f6b73" stroke-linecap="round" stroke-linejoin="round"/>
  <polyline id="kFillS" fill="none" stroke="#ECECEC" stroke-linecap="round" stroke-linejoin="round"/>
  <circle id="kKnee" fill="#5f6b73"/>
  <circle id="kShoe" fill="#222"/>
  <line id="vec" stroke="#1D9E75" stroke-width="4"/>
  <circle id="vhead" r="5" fill="#1D9E75"/>
</svg>

<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0;">
  <button id="btn" style="padding:6px 14px;border-radius:6px;border:1px solid #bbb;background:#f5f5f5;cursor:pointer;">▶ Animar patada</button>
  <button id="rst" style="padding:6px 14px;border-radius:6px;border:1px solid #bbb;background:#f5f5f5;cursor:pointer;">↺ Reiniciar</button>
  <span id="tt" style="font-size:14px;">t = 0.000 s</span>
  <span id="estado" style="font-size:14px;color:#666;"></span>
</div>

<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-bottom:8px;">
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">v punta (m/s)</div><div id="mv" style="font-size:20px;font-weight:500;">0.00</div></div>
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">K pierna (J)</div><div id="mk" style="font-size:20px;font-weight:500;">0.00</div></div>
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">p centro de masa (kg·m/s)</div><div id="mp" style="font-size:20px;font-weight:500;">0.00</div></div>
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">Ángulo cadera (°)</div><div id="mt" style="font-size:20px;font-weight:500;">0.0</div></div>
</div>
<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-bottom:10px;">
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">ω (rad/s)</div><div id="mw" style="font-size:20px;font-weight:500;">0.00</div></div>
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">α (rad/s²)</div><div id="ma" style="font-size:20px;font-weight:500;">0.00</div></div>
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">τ (N·m)</div><div id="mtau" style="font-size:20px;font-weight:500;">0.00</div></div>
  <div style="background:#f5f5f5;border-radius:8px;padding:8px;"><div style="font-size:12px;color:#666;">Flexión de rodilla (°)</div><div id="mb" style="font-size:20px;font-weight:500;">0.0</div></div>
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
const $ = id => document.getElementById(id);
const GY = 380, hx = 260;
const u = D.L * Math.min(180, 300 / D.L);
const hy = GY - 0.95 * u;               // altura de la cadera
const rel = (x, y) => [hx + x * u, hy + y * u];
const ps = a => a.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");

function poly(o, f, pts, w) {
  $(o).setAttribute("points", ps(pts)); $(o).setAttribute("stroke-width", (w * 1.25).toFixed(1));
  $(f).setAttribute("points", ps(pts)); $(f).setAttribute("stroke-width", w.toFixed(1));
}

// ----- Figura fija -----
$("vista").textContent = D.vista + " · " + D.nombre;
poly("sOut", "sFill", [[hx, hy], [hx - 0.12 * u, hy + 0.5 * u], [hx - 0.1 * u, GY - 0.05 * u]], 0.17 * u);
$("sShoe").setAttribute("cx", hx - 0.06 * u); $("sShoe").setAttribute("cy", GY - 0.04 * u);
$("sShoe").setAttribute("rx", 0.14 * u); $("sShoe").setAttribute("ry", 0.05 * u);

poly("aOutD", "aFillD", [rel(-0.18, -0.56), rel(-0.24, -0.3), rel(-0.25, -0.1)], 0.09 * u);
poly("aOutG", "aFillG", [rel(0.18, -0.56), rel(0.3, -0.36), rel(0.14, -0.66)], 0.09 * u);

const nk = $("neck"), n1 = rel(0, -0.6), n2 = rel(0, -0.72);
nk.setAttribute("x1", n1[0]); nk.setAttribute("y1", n1[1]);
nk.setAttribute("x2", n2[0]); nk.setAttribute("y2", n2[1]);
nk.setAttribute("stroke-width", (0.1 * u).toFixed(1));

$("jacket").setAttribute("points", ps([rel(-0.2, -0.62), rel(0.2, -0.62), rel(0.17, -0.04), rel(-0.17, -0.04)]));
const b1 = rel(-0.17, -0.06), b2 = rel(0.17, -0.06), bl = $("belt");
bl.setAttribute("x1", b1[0]); bl.setAttribute("y1", b1[1]);
bl.setAttribute("x2", b2[0]); bl.setAttribute("y2", b2[1]);
bl.setAttribute("stroke-width", (0.08 * u).toFixed(1));

$("hair").setAttribute("cx", hx); $("hair").setAttribute("cy", hy - 0.88 * u);
$("hair").setAttribute("rx", 0.14 * u); $("hair").setAttribute("ry", 0.11 * u);
$("head").setAttribute("cx", hx); $("head").setAttribute("cy", hy - 0.84 * u);
$("head").setAttribute("r", 0.13 * u);
$("eye").setAttribute("cx", hx + 0.06 * u); $("eye").setAttribute("cy", hy - 0.85 * u);
$("eye").setAttribute("r", 0.015 * u);

// ----- Trayectoria envolvente (punta de la pierna recta) -----
let arc = "";
for (let i = 0; i <= 60; i++) {
  const a = D.sweep * i / 60;
  const x = hx + u * D.dir * Math.sin(a), y = hy + u * Math.cos(a);
  arc += (i ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1) + " ";
}
$("arc").setAttribute("d", arc);

// ----- Geometría de la pierna que patea -----
// th: ángulo del muslo medido desde la vertical hacia abajo, con sentido D.dir
// La rodilla se flexiona al inicio y se extiende hacia el impacto
function legAt(i) {
  const th = D.th[i];
  const s = Math.min(Math.max(th / D.sweep, 0), 1);
  const beta = D.pliegue * Math.pow(1 - s, 2);
  const d = D.dir;
  const kn = [hx + 0.5 * u * d * Math.sin(th), hy + 0.5 * u * Math.cos(th)];
  const sa = th - beta * d;
  const ft = [kn[0] + 0.5 * u * d * Math.sin(sa), kn[1] + 0.5 * u * Math.cos(sa)];
  return { hip: [hx, hy], kn, ft, s, beta };
}

// ----- Gráficas -----
const vmax = Math.max(...D.v) * 1.1 || 1;
const kmax = Math.max(...D.k) * 1.1 || 1;
$("lv").textContent = "v máx " + Math.max(...D.v).toFixed(1) + " m/s";
$("lk").textContent = "K máx " + Math.max(...D.k).toFixed(1) + " J";
function gpoly(arr, max) {
  return D.t.map((t, i) =>
    (10 + 580 * t / D.tend).toFixed(1) + "," + (130 - 120 * arr[i] / max).toFixed(1)
  ).join(" ");
}
$("pv").setAttribute("points", gpoly(D.v, vmax));
$("pk").setAttribute("points", gpoly(D.k, kmax));

// ----- Cuadro -----
function frame(i) {
  const th = D.th[i], t = D.t[i];
  const g = legAt(i);
  poly("kOutT", "kFillT", [g.hip, g.kn], 0.2 * u);
  poly("kOutS", "kFillS", [g.kn, g.ft], 0.16 * u);
  $("kKnee").setAttribute("cx", g.kn[0]); $("kKnee").setAttribute("cy", g.kn[1]);
  $("kKnee").setAttribute("r", 0.05 * u);
  $("kShoe").setAttribute("cx", g.ft[0]); $("kShoe").setAttribute("cy", g.ft[1]);
  $("kShoe").setAttribute("r", 0.08 * u);

  // Vector velocidad tangente a la trayectoria de la punta
  const d = D.dir, len = 4 * D.v[i];
  const vx = g.ft[0] + d * Math.cos(th) * len, vy = g.ft[1] - Math.sin(th) * len;
  $("vec").setAttribute("x1", g.ft[0]); $("vec").setAttribute("y1", g.ft[1]);
  $("vec").setAttribute("x2", vx); $("vec").setAttribute("y2", vy);
  $("vhead").setAttribute("cx", vx); $("vhead").setAttribute("cy", vy);

  // Torso: se inclina en sentido opuesto a la patada
  const lean = -d * D.lean * g.s;
  $("upper").setAttribute("transform", "rotate(" + lean.toFixed(2) + " " + hx + " " + hy.toFixed(1) + ")");

  // Rastro del pie
  let r = "";
  for (let j = 0; j <= i; j++) {
    const gj = legAt(j);
    r += (j ? "L" : "M") + gj.ft[0].toFixed(1) + "," + gj.ft[1].toFixed(1) + " ";
  }
  $("rastro").setAttribute("d", r);

  $("tt").textContent = "t = " + t.toFixed(3) + " s";
  $("mv").textContent = D.v[i].toFixed(2);
  $("mk").textContent = D.k[i].toFixed(2);
  $("mp").textContent = D.p[i].toFixed(2);
  $("mt").textContent = (th * 180 / Math.PI).toFixed(1);
  $("mw").textContent = D.w[i].toFixed(2);
  $("ma").textContent = D.al[i].toFixed(2);
  $("mtau").textContent = D.tau[i].toFixed(2);
  $("mb").textContent = (g.beta * 180 / Math.PI).toFixed(1);
  const cx = 10 + 580 * t / D.tend;
  $("cur").setAttribute("x1", cx); $("cur").setAttribute("x2", cx);
}

// ----- Reproducción escalada por "cámara lenta" -----
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
    $("btn").textContent = "▶ Repetir";
    $("estado").textContent = "Impacto";
    return;
  }
  requestAnimationFrame(n => loop(id, n));
}

$("btn").onclick = () => {
  if (playing) {
    playing = false; last = null;
    $("btn").textContent = "▶ Continuar";
    $("estado").textContent = "En pausa";
    return;
  }
  if (elapsed >= dur) elapsed = 0;
  playing = true; last = null; loopId++;
  $("btn").textContent = "⏸ Pausa";
  $("estado").textContent = "Reproduciendo";
  requestAnimationFrame(n => loop(loopId, n));
};

$("rst").onclick = () => {
  playing = false; last = null; elapsed = 0; loopId++;
  $("btn").textContent = "▶ Animar patada";
  $("estado").textContent = "";
  frame(0);
};

frame(0);
</script>
"""

# ---------- Pestañas ----------
tab_sim, tab_datos, tab_teoria = st.tabs(["Simulación", "Datos", "Teoría"])

with tab_teoria:
    st.subheader("Física detrás de la patada")
    st.markdown("La pierna se modela como una barra uniforme que gira alrededor de la cadera.")
    st.latex(r"I = \tfrac{1}{3} m L^2 \qquad \text{(respecto al extremo)}")
    st.latex(r"v_{punta} = \omega L \qquad v_{cm} = \tfrac{1}{2}\omega L")
    st.latex(r"K = \tfrac{1}{2} I \omega^2 \qquad p = m\,v_{cm}")
    st.latex(r"\alpha = \frac{\omega_{max}}{t_{acc}} \qquad \theta_{acc} = \tfrac{1}{2}\omega_{max}t_{acc}")
    st.latex(r"\tau = I\,\alpha \qquad \Delta p = m_{eff}\,v_{punta} \qquad F_{prom} = \frac{\Delta p}{\Delta t_{contacto}}")
    st.markdown(
        "La velocidad de la punta crece con la longitud de la pierna y la velocidad angular. "
        "Como la energía cinética depende de ω², pequeños aumentos de velocidad angular "
        "producen grandes aumentos de energía. Un contacto más corto aumenta la fuerza media."
    )
    st.markdown("**Supuestos por técnica** (ilustrativos, no mediciones):")
    st.dataframe(pd.DataFrame([
        {
            "Técnica": k,
            "Plano": v["vista"],
            "Barrido (°)": v["sweep"],
            "ω máx (rad/s)": v["w_max"],
            "t aceleración (s)": v["t_acc"],
            "Flexión inicial (°)": v["pliegue"],
            "Sentido": "adelante" if v["dir"] == 1 else "atrás",
            "Descripción": v["descr"],
        }
        for k, v in TECNICAS.items()
    ]), hide_index=True, use_container_width=True)

sweep = np.radians(P["sweep"])
res = simular(L, m_pierna, P["w_max"], P["t_acc"], sweep)

with tab_sim:
    st.subheader(tecnica)
    st.write(P["descr"])
    if res is None:
        st.error("El tiempo de aceleración es demasiado largo para el ángulo de barrido de esta técnica.")
    else:
        imp = impacto(res, m_eff, dt_contacto)
        if velocidad_valida(imp["v"], "Velocidad de la punta al impacto"):
            idx = np.linspace(0, len(res["t"]) - 1, 120).astype(int)
            datos = {
                "nombre": tecnica,
                "vista": P["vista"],
                "L": float(L),
                "dir": P["dir"],
                "lean": float(P["lean"]),
                "pliegue": float(np.radians(P["pliegue"])),
                "sweep": float(sweep),
                "tend": float(res["t_imp"]),
                "slow": int(lento),
                "t": res["t"][idx].round(4).tolist(),
                "th": res["theta"][idx].round(4).tolist(),
                "w": res["omega"][idx].round(3).tolist(),
                "al": res["alpha"][idx].round(3).tolist(),
                "tau": res["tau"][idx].round(3).tolist(),
                "v": res["v"][idx].round(3).tolist(),
                "k": res["k"][idx].round(3).tolist(),
                "p": res["p"][idx].round(3).tolist(),
            }
            components.html(PLANTILLA.replace("__DATA__", json.dumps(datos)), height=900)

            st.subheader("Resultados en el impacto")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Velocidad de la punta (m/s)", f"{imp['v']:.2f}")
            r2.metric("Cantidad de movimiento transferida (kg·m/s)", f"{imp['p']:.2f}")
            r3.metric("Fuerza media en el contacto (N)", f"{imp['F']:.0f}")
            r4.metric("Energía cinética de la pierna (J)", f"{res['k'][-1]:.1f}")

with tab_datos:
    st.subheader("Datos de la patada seleccionada")
    if res is not None:
        df = pd.DataFrame({
            "ω (rad/s)": res["omega"],
            "α (rad/s²)": res["alpha"],
            "τ (N·m)": res["tau"],
            "v punta (m/s)": res["v"],
            "K pierna (J)": res["k"],
            "p cm (kg·m/s)": res["p"],
        }, index=np.round(res["t"], 3))
        df.index.name = "t (s)"
        st.line_chart(df[["ω (rad/s)", "α (rad/s²)", "τ (N·m)"]])
        st.line_chart(df[["v punta (m/s)", "K pierna (J)", "p cm (kg·m/s)"]])

        st.subheader("Comparación de las cuatro técnicas")
        filas = []
        for nombre, Q in TECNICAS.items():
            r = simular(L, m_pierna, Q["w_max"], Q["t_acc"], np.radians(Q["sweep"]))
            if r is None:
                continue
            im = impacto(r, m_eff, dt_contacto)
            filas.append({
                "Técnica": nombre,
                "Tiempo al impacto (s)": round(r["t_imp"], 3),
                "ω máx (rad/s)": round(float(r["omega"].max()), 2),
                "α máx (rad/s²)": round(float(r["alpha"].max()), 2),
                "τ máx (N·m)": round(float(r["tau"].max()), 2),
                "v punta (m/s)": round(im["v"], 2),
                "p transferida (kg·m/s)": round(im["p"], 2),
                "F media (N)": round(im["F"], 0),
                "K pierna (J)": round(float(r["k"][-1]), 1),
            })
        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
        st.caption("Todos los valores son del modelo con los supuestos de la tabla de la pestaña Teoría.")
