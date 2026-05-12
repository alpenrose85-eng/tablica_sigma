import math
import streamlit as st
import pandas as pd

# --- Model constants ---
P = 0.53
M = 0.81

# Known grain sizes (mm) for grain numbers G
_KNOWN_DG = {3: 0.125, 5: 0.062, 8: 0.022, 9: 0.015, 10: 0.011}


def grain_size(G: int) -> float:
    """Return grain size dg (mm) for grain number G=3..10."""
    if G in _KNOWN_DG:
        return _KNOWN_DG[G]
    # Linear interpolation between adjacent known points
    known_G = sorted(_KNOWN_DG)
    for i in range(len(known_G) - 1):
        g0, g1 = known_G[i], known_G[i + 1]
        if g0 < G < g1:
            t = (G - g0) / (g1 - g0)
            return _KNOWN_DG[g0] + t * (_KNOWN_DG[g1] - _KNOWN_DG[g0])
    raise ValueError(f"G={G} вне диапазона 3..10")


def compute_A(dg: float) -> float:
    """A(dg) = exp(-3.8 + 0.63·ln(dg) + 0.22·ln(dg)²)"""
    ln_dg = math.log(dg)
    return math.exp(-3.8 + 0.63 * ln_dg + 0.22 * ln_dg ** 2)


def compute_temperature(c_sigma: float, dg: float, tau: float) -> float:
    """T = 550 + 350 · (c_sigma / (A(dg) · tau^p))^(1/m)"""
    A = compute_A(dg)
    return 550 + 350 * (c_sigma / (A * tau ** P)) ** (1 / M)


def compute_k1(G: int) -> float:
    """k1 = 0.002·G^3 - 0.0237·G^2 + 0.1007·G + 0.3419"""
    return 0.002 * G ** 3 - 0.0237 * G ** 2 + 0.1007 * G + 0.3419


def compute_k2(k1: float) -> float:
    """k2 = 220.5 / exp(-k1 · 14.698)"""
    return 220.5 / math.exp(-k1 * 14.698)


def compute_tau_ost(sigma: float, temperature_c: float, G: int) -> float:
    """Solve for τ_ост from:
    σ = (k2 / 1.25) · exp(-k1 · T · (log10(τ_ост) - 2·log10(T) + 19.52) · 10^-3)
    where σ is the user-provided stress.

    For this formula T is used in Kelvin.
    """
    temperature_k = temperature_c + 273.15
    k1 = compute_k1(G)
    k2 = compute_k2(k1)
    sigma_ratio = sigma / (k2 / 1.25)
    if sigma_ratio <= 0:
        raise ValueError("Некорректное значение σ для расчёта τ_ост")

    exponent_term = -math.log(sigma_ratio) / (k1 * temperature_k * 1e-3)
    log_tau_ost = exponent_term + 2 * math.log10(temperature_k) - 19.52
    return 10 ** log_tau_ost


def build_table(tau: float, c_sigma_list: list[float], sigma: float) -> pd.DataFrame:
    grain_numbers = list(range(3, 11))
    parameter_rows = ["dg, мм"]
    for c in c_sigma_list:
        parameter_rows.append(f"T при c_σ = {c:.2f}%")
        parameter_rows.append(f"τ_ост при c_σ = {c:.2f}% (×10⁵ ч)")

    data = {"Параметр": parameter_rows}

    for G in grain_numbers:
        dg = grain_size(G)
        column_values = [f"{dg:.3f}"]
        for c in c_sigma_list:
            temperature = compute_temperature(c, dg, tau)
            tau_ost = compute_tau_ost(sigma, temperature, G)
            tau_ost_hours = round(tau_ost)
            tau_ost_hundred_thousands = tau_ost_hours / 100000
            column_values.append(f"{round(temperature):.0f}")
            column_values.append(f"{tau_ost_hundred_thousands:.2f}")
        data[f"G = {G}"] = column_values

    return pd.DataFrame(data)


# --- Streamlit UI ---
st.set_page_config(page_title="Температура σ-фазы", layout="centered")
st.title("Расчёт температуры образования σ-фазы")

with st.expander("Описание модели", expanded=False):
    st.markdown(
        r"""
Модель позволяет определить температуру $T$ (°C), при которой за время $\tau$ (ч)
образуется заданное количество σ-фазы $c_\sigma$ (%) для стали с зерном номера $G$.

**Вспомогательные выражения:**

$$A(d_g) = \exp\!\bigl(-3{,}8 + 0{,}63\ln d_g + 0{,}22(\ln d_g)^2\bigr)$$

**Основная формула:**

$$c_\sigma = A(d_g)\cdot\tau^{p}\cdot\left(\frac{T-550}{350}\right)^{m}$$

где $p = 0{,}53$, $m = 0{,}81$.

**Решение относительно температуры:**

$$T = 550 + 350\cdot\left(\frac{c_\sigma}{A(d_g)\cdot\tau^{p}}\right)^{1/m}$$

**Дополнительный расчёт остаточного времени:**

Здесь $\sigma$ — это **напряжение, которое вводит пользователь**, а в формуле для $\tau_{ост}$ температура $T$ подставляется в **Кельвинах**.

$$\sigma = \frac{k_2}{1{,}25}\exp\left(-k_1 T\left(\log \tau_{\text{ост}} - 2\log T + 19{,}52\right)\cdot10^{-3}\right)$$

$$k_1 = 0{,}002G^3 - 0{,}0237G^2 + 0{,}1007G + 0{,}3419$$

$$k_2 = \frac{220{,}5}{\exp(-k_1\cdot14{,}698)}$$

Размер зерна $d_g$ (мм) задаётся номером $G$ по ГОСТ/ASTM; для промежуточных номеров
используется линейная интерполяция.
""",
        unsafe_allow_html=False,
    )

st.subheader("Параметры расчёта")

tau = st.number_input(
    "Время τ, ч",
    min_value=0.0,
    value=10.0,
    step=1.0,
    format="%.2f",
)

sigma_input = st.number_input(
    "Напряжение σ, МПа",
    min_value=0.0,
    value=100.0,
    step=1.0,
    format="%.2f",
)

st.markdown("**Содержание σ-фазы c_σ (%)**")
st.caption("Добавьте одно или несколько значений — таблица расширится автоматически.")

# Dynamic list of c_sigma values stored in session_state
if "c_sigma_list" not in st.session_state:
    st.session_state.c_sigma_list = [5.0]

cols = st.columns([3, 1])
new_c = cols[0].number_input(
    "Новое значение c_σ (%)",
    min_value=0.0,
    value=5.0,
    step=1.0,
    format="%.2f",
    label_visibility="collapsed",
)
if cols[1].button("Добавить"):
    if new_c <= 0:
        st.warning("Значение c_σ должно быть положительным.")
    elif new_c in st.session_state.c_sigma_list:
        st.warning(f"Значение {new_c:.2f}% уже добавлено.")
    else:
        st.session_state.c_sigma_list.append(new_c)
        st.session_state.c_sigma_list.sort()

if st.session_state.c_sigma_list:
    selected = st.multiselect(
        "Выбранные значения c_σ (снимите галочку, чтобы удалить):",
        options=sorted(st.session_state.c_sigma_list),
        default=sorted(st.session_state.c_sigma_list),
        format_func=lambda x: f"{x:.2f}%",
    )
    st.session_state.c_sigma_list = sorted(selected)

st.divider()

# Validation and calculation
errors = []
if tau <= 0:
    errors.append("Время τ должно быть положительным числом (> 0).")
if sigma_input <= 0:
    errors.append("Напряжение σ должно быть положительным числом (> 0).")
if not st.session_state.c_sigma_list:
    errors.append("Добавьте хотя бы одно значение c_σ.")

if errors:
    for e in errors:
        st.error(e)
else:
    st.subheader("Таблица температур T, °C")
    df = build_table(tau, st.session_state.c_sigma_list, sigma_input)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        f"τ = {tau:.2f} ч · σ = {sigma_input:.2f} МПа · p = {P} · m = {M} · "
        "T в градусах Цельсия · dg в мм · τ_ост в сотнях тысяч часов"
    )
