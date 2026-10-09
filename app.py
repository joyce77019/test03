"""
拓扑霍尔效应 (THE) 数据分析工具 —— 版本 0.2

本版本在“数据导入与诊断”基础上，新增：
1. ETO 双通道对齐：Resistance Ch1 (Ohms) -> Rxx，Resistance Ch2 (Ohms) -> Rxy，
   两个通道交替读数，按 (扫描方向, 磁场) 合并到同一磁场网格；
2. 单温度预处理与平滑：Savitzky-Golay 平滑，原始/平滑曲线叠加对比；
3. 零点校正：自动估算 H≈0 偏移并支持手动微调；
4. 对称化：选定升场/降场分支，计算 Rxx_sym 与 Rxy_anti。

运行方式：
    streamlit run app.py
"""

import hashlib
import io
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import linregress


st.set_page_config(
    page_title="拓扑霍尔效应 (THE) 数据分析工具",
    page_icon="🧲",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# 数据解析辅助函数
# --------------------------------------------------------------------------- #

SEPARATORS = [",", "\t", ";", r"\s+"]

TEMPERATURE_KEYWORDS = ["temperature", "temp", "kelvin", "t(k)", "t (k)"]
FIELD_KEYWORDS = [
    "magnetic field",
    "field",
    "h(oe)",
    "h (oe)",
    "h(t)",
    "h (t)",
    "h_oe",
    "h_t",
]
RXX_KEYWORDS = [
    "ch1",
    "channel 1",
    "ch 1",
    "rxx",
    "longitudinal",
    "xx",
]
RXY_KEYWORDS = [
    "ch2",
    "channel 2",
    "ch 2",
    "rxy",
    "ryx",
    "hall",
    "transverse",
    "xy",
]
MOMENT_KEYWORDS = ["magnetization", "moment", "m(emu)", "m (emu)", "long moment"]


# 每个温度的默认参数。首次选择某个温度时，用这份默认值初始化其 params。
DEFAULT_PARAMS = {
    "branch": "升场",
    "rxx_window": 11,
    "rxx_order": 2,
    "rxy_window": 11,
    "rxy_order": 2,
    "spline_s": 0.0,
    "zero_delta": 0.0,
    "remove_outliers": False,
    "outlier_sigma": 5.0,
    "use_h_range": False,
    "h_min": 0.0,
    "h_max": 0.0,
    "h_range_initialized": False,
    "use_fit_x_range": False,
    "fit_x_min": 0.0,
    "fit_x_max": 0.0,
}


# 样品尺寸默认值与侧边栏控件固定 key（供会话恢复写回使用）。
SAMPLE_DEFAULTS = {"width": 1.0, "thickness": 0.1, "length": 1.0}

KEY_WIDTH = "sample_width"
KEY_THICKNESS = "sample_thickness"
KEY_LENGTH = "sample_length"
KEY_VSM_FRACTION = "vsm_min_fraction"
KEY_ETO_UPLOADER = "eto_uploader"
KEY_VSM_UPLOADER = "vsm_uploader"


def _decode(data: bytes) -> str:
    """用多种编码尝试解码文件内容，尽量兼容中文/西文设备导出文件。"""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _try_read(text: str, sep: str, header: int):
    """尝试用给定分隔符和表头行号读取数据。失败时返回 None。"""
    try:
        return pd.read_csv(
            io.StringIO(text),
            sep=sep,
            header=header,
            engine="python",
            on_bad_lines="skip",
            skip_blank_lines=True,
        )
    except Exception:
        return None


def _score_dataframe(df) -> float:
    """给一个候选 DataFrame 打分，数值列越多、表头含物理关键字越多则分越高。"""
    if df is None or df.shape[1] < 2 or len(df) < 2:
        return -1.0

    numeric_columns = 0
    numeric_cells = 0
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() > 0.5:
            numeric_columns += 1
        numeric_cells += int(numeric.notna().sum())

    if numeric_columns < 2:
        return -1.0

    header_text = " ".join(str(c).lower() for c in df.columns)
    keyword_hits = 0
    all_keywords = (
        TEMPERATURE_KEYWORDS
        + FIELD_KEYWORDS
        + RXX_KEYWORDS
        + RXY_KEYWORDS
        + MOMENT_KEYWORDS
    )
    for keyword in all_keywords:
        if keyword in header_text:
            keyword_hits += 1

    return numeric_columns * 10.0 + numeric_cells + keyword_hits * 100.0


def read_data_file(data: bytes, filename: str):
    """自动识别分隔符与表头，返回解析后的 DataFrame。"""
    text = _decode(data)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    best_df = None
    best_score = -1.0

    max_header_rows = min(40, len(lines))
    for sep in SEPARATORS:
        for header in range(max_header_rows):
            df = _try_read(text, sep, header)
            score = _score_dataframe(df)
            if score > best_score:
                best_df = df
                best_score = score

    return best_df


def _normalize(name) -> str:
    """去掉多余空白并转小写，用于稳定的列名匹配。"""
    return " ".join(str(name).split()).lower()


def find_column(df, keywords):
    """返回第一个名称包含任意关键字的列名；找不到返回 None。"""
    if df is None:
        return None
    for col in df.columns:
        normalized = _normalize(col)
        for keyword in keywords:
            if keyword in normalized:
                return col
    return None


def split_by_temperature(df):
    """按“温度”列把 DataFrame 拆成 {温度: 子DataFrame} 的字典。"""
    if df is None:
        return None, {}

    temp_col = find_column(df, TEMPERATURE_KEYWORDS)
    if temp_col is None:
        return None, {}

    numeric_temps = pd.to_numeric(df[temp_col], errors="coerce")
    if numeric_temps.notna().sum() < 1:
        return temp_col, {}

    work = df.copy()
    # 温度先四舍五入到整数，再按整数温度分组（同时消除浮点噪声）。
    work["__temp_key__"] = np.floor(numeric_temps + 0.5)

    groups = OrderedDict()
    for temp_key, group in work.groupby("__temp_key__", dropna=True):
        groups[float(temp_key)] = group.drop(columns=["__temp_key__"])

    return temp_col, groups


def filter_temperatures_by_count(groups, min_fraction=0.8):
    """
    只保留数据点数不低于 min_fraction * 最大点数的温度。

    返回 (筛选后的 groups, 各温度点数统计)。
    """
    if not groups:
        return {}, {}

    counts = {temp: len(group) for temp, group in groups.items()}
    max_count = max(counts.values())
    threshold = max(1, int(round(max_count * min_fraction)))
    kept = {
        temp: group
        for temp, group in groups.items()
        if counts[temp] >= threshold
    }
    return kept, counts


def compute_zero_field_offset(field, signal):
    """估算 H≈0 附近的信号值，作为零点偏移。"""
    field = pd.to_numeric(field, errors="coerce")
    signal = pd.to_numeric(signal, errors="coerce")
    valid = field.notna() & signal.notna()
    field = field[valid].to_numpy(dtype=float)
    signal = signal[valid].to_numpy(dtype=float)

    if len(field) == 0:
        return None

    span = max(abs(field.min()), abs(field.max()))
    threshold = max(span * 0.02, 1e-12)
    near_zero = np.abs(field) <= threshold

    if near_zero.any():
        return float(np.mean(signal[near_zero]))

    closest = int(np.argmin(np.abs(field)))
    return float(signal[closest])


@st.cache_data(show_spinner=False)
def parse_uploaded_file(data: bytes, filename: str):
    """缓存解析结果，避免每次界面刷新都重新解析文件。"""
    return read_data_file(data, filename)


# --------------------------------------------------------------------------- #
# ETO 双通道对齐 / 平滑 / 对称化
# --------------------------------------------------------------------------- #

def assign_sweep_direction(field_series):
    """根据磁场随行号的变化方向，给每一行标注扫描方向（1 升场，-1 降场）。"""
    h = pd.to_numeric(field_series, errors="coerce").to_numpy(dtype=float)
    direction = np.zeros(len(h), dtype=int)
    previous = 0
    for i in range(1, len(h)):
        diff = h[i] - h[i - 1]
        if diff > 0:
            previous = 1
        elif diff < 0:
            previous = -1
        direction[i] = previous
    if len(direction) > 1 and direction[0] == 0:
        direction[0] = direction[1] if direction[1] != 0 else 0
    return direction


def extract_channel_series(group, field_col, value_col, scale=1.0, field_scale=1.0):
    """从某个通道提取 (H, value, branch)，按通道去除空值，可乘换算因子。"""
    if group is None or field_col is None or value_col is None:
        return pd.DataFrame(columns=["H", "value", "branch"])

    h = pd.to_numeric(group[field_col], errors="coerce")
    v = pd.to_numeric(group[value_col], errors="coerce")
    valid = h.notna() & v.notna()
    if not valid.any():
        return pd.DataFrame(columns=["H", "value", "branch"])

    direction = assign_sweep_direction(group[field_col])
    valid_index = np.flatnonzero(valid.to_numpy())
    return pd.DataFrame(
        {
            "H": h[valid].to_numpy(dtype=float) * field_scale,
            "value": v[valid].to_numpy(dtype=float) * scale,
            "branch": np.where(direction[valid_index] >= 0, "up", "down"),
        }
    )


def choose_branch_series(series, branch_choice):
    """按用户选择返回单一升场/降场分支的通道序列。"""
    if series is None or series.empty:
        return series, None, []

    available = sorted(series["branch"].unique().tolist())
    branch_key = {"升场": "up", "降场": "down", "自动": None}.get(branch_choice)

    if branch_key is None:
        return series.copy(), "自动(全部)", available

    selected = series[series["branch"] == branch_key].copy()
    if selected.empty:
        return series.copy(), f"{branch_choice}(回退为全部)", available
    return selected, branch_choice, available


def smooth_y(y, window, polyorder):
    """Savitzky-Golay 平滑；自动处理奇数窗口与阶数边界。"""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y

    window = int(window)
    if window % 2 == 0:
        window += 1
    window = max(3, min(window, len(y)))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y

    polyorder = int(polyorder)
    polyorder = max(0, min(polyorder, window - 1))
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def smooth_series_df(series, window, polyorder):
    """对单个通道序列按 H 升序平滑，返回带 smooth 列的 DataFrame。"""
    if series is None or series.empty:
        return None
    d = series.sort_values("H").reset_index(drop=True).copy()
    d["smooth"] = smooth_y(d["value"].to_numpy(dtype=float), window, polyorder)
    return d


def _dedup_series(h, v):
    """按相同磁场取平均，保证 UnivariateSpline 的 x 严格递增。"""
    frame = pd.DataFrame(
        {"H": np.asarray(h, dtype=float), "V": np.asarray(v, dtype=float)}
    )
    frame = frame.groupby("H", as_index=False)["V"].mean().sort_values("H")
    return frame["H"].to_numpy(dtype=float), frame["V"].to_numpy(dtype=float)


def filter_h_range(series, h_min, h_max):
    """按磁场范围过滤通道序列。"""
    if series is None or series.empty:
        return series
    return series[(series["H"] >= h_min) & (series["H"] <= h_max)].copy()


def remove_outliers(series, window, polyorder, sigma):
    """
    用 Savitzky-Golay 平滑作为基线，剔除偏离基线超过 sigma * MAD 的坏点。

    返回 (清洗后的序列, 剔除点数)。
    """
    if series is None or series.empty:
        return series, 0

    d = series.sort_values("H").reset_index(drop=True).copy()
    y = d["value"].to_numpy(dtype=float)
    if len(y) < max(5, int(window)):
        return d, 0

    baseline = smooth_y(y, window, polyorder)
    residual = y - baseline
    center = np.median(residual)
    mad = np.median(np.abs(residual - center))
    if mad == 0:
        mad = np.median(np.abs(y - np.median(y)))
    if mad == 0:
        return d, 0

    threshold = sigma * 1.4826 * mad
    keep = np.abs(residual - center) <= threshold
    removed = int((~keep).sum())
    return d[keep].reset_index(drop=True), removed


def prepare_eto_series(
    series, branch_choice, h_range, remove_flag, sigma, window, polyorder
):
    """对 ETO 单通道序列执行：选分支、选范围、去坏点。"""
    branch, label, available = choose_branch_series(series, branch_choice)
    if branch is None or branch.empty:
        return branch, label, available, 0

    if h_range is not None:
        branch = filter_h_range(branch, *h_range)

    removed = 0
    if remove_flag and not branch.empty:
        branch, removed = remove_outliers(branch, window, polyorder, sigma)
    return branch, label, available, removed


def prepare_vsm_series(series, h_range, remove_flag, sigma):
    """对 VSM 序列执行：选范围、去坏点（不区分扫描方向）。"""
    if series is None or series.empty:
        return series, 0

    d = series.sort_values("H").reset_index(drop=True).copy()
    if h_range is not None:
        d = filter_h_range(d, *h_range)

    removed = 0
    if remove_flag and not d.empty:
        # VSM 去坏点使用固定的稳健基线（窗口 11、二阶）。
        d, removed = remove_outliers(d, 11, 2, sigma)
    return d, removed


def symmetrize_channels(
    rxx_series,
    rxy_series,
    rxx_window,
    rxx_order,
    rxy_window,
    rxy_order,
    offset,
    spline_s,
):
    """
    对单一扫描分支分别插值 Rxx/Rxy 到对称网格，再计算 Rxx_sym 与 Rxy_anti。

    返回 DataFrame：H(≥0), Rxx_sym, Rxy_anti；无法处理时返回 None。
    """
    if (
        rxx_series is None
        or rxy_series is None
        or rxx_series.empty
        or rxy_series.empty
    ):
        return None

    rxx_series = rxx_series.sort_values("H")
    rxy_series = rxy_series.sort_values("H")

    rxx_smooth = smooth_y(
        rxx_series["value"].to_numpy(dtype=float), rxx_window, rxx_order
    )
    rxy_smooth = (
        smooth_y(rxy_series["value"].to_numpy(dtype=float), rxy_window, rxy_order)
        - offset
    )

    h_rxx, v_rxx = _dedup_series(
        rxx_series["H"].to_numpy(dtype=float), rxx_smooth
    )
    h_rxy, v_rxy = _dedup_series(
        rxy_series["H"].to_numpy(dtype=float), rxy_smooth
    )

    if len(h_rxx) < 4 or len(h_rxy) < 4:
        return None

    h_negative = max(h_rxx.min(), h_rxy.min())
    h_positive = min(h_rxx.max(), h_rxy.max())
    if h_negative >= 0 or h_positive <= 0:
        return None

    h_limit = min(abs(h_negative), abs(h_positive))
    n_points = 600
    grid = np.linspace(-h_limit, h_limit, n_points)

    try:
        spline_rxx = UnivariateSpline(h_rxx, v_rxx, k=3, s=spline_s)
        spline_rxy = UnivariateSpline(h_rxy, v_rxy, k=3, s=spline_s)
        rxx_interp = spline_rxx(grid)
        rxy_interp = spline_rxy(grid)
    except Exception:
        return None

    half = n_points // 2
    negative_reversed = slice(half - 1, None, -1)
    rxx_sym = (rxx_interp[half:] + rxx_interp[negative_reversed]) / 2.0
    rxy_anti = (rxy_interp[half:] - rxy_interp[negative_reversed]) / 2.0

    return pd.DataFrame(
        {
            "H": grid[half:],
            "Rxx_sym": rxx_sym,
            "Rxy_anti": rxy_anti,
        }
    )


def interpolate_vsm_to_grid(vsm_series, grid, spline_s=0.0):
    """把 VSM 磁矩 M(H) 插值到给定磁场网格，返回插值后的磁矩数组。"""
    if vsm_series is None or vsm_series.empty:
        return None

    d = vsm_series.sort_values("H").reset_index(drop=True)
    h, v = _dedup_series(
        d["H"].to_numpy(dtype=float), d["value"].to_numpy(dtype=float)
    )
    if len(h) < 4:
        return None

    try:
        spline = UnivariateSpline(h, v, k=3, s=spline_s)
        return spline(np.asarray(grid, dtype=float))
    except Exception:
        return None


def ensure_temp_params(temp):
    """确保 session_state 中该温度已有 params/results 槽位。"""
    if "temp_data" not in st.session_state:
        st.session_state.temp_data = {}

    data = st.session_state.temp_data
    if temp not in data:
        data[temp] = {"params": dict(DEFAULT_PARAMS), "results": None}
    if "params" not in data[temp]:
        data[temp]["params"] = dict(DEFAULT_PARAMS)
    if "results" not in data[temp]:
        data[temp]["results"] = None
    return data[temp]["params"]


def serialize_session(temp_data):
    """把 session 里的 temp_data 转成可 JSON 序列化的普通 dict。"""
    out = {}
    for temp, entry in temp_data.items():
        params = entry.get("params", {})
        result = entry.get("results")
        serialized_result = None
        if result:
            table = result.get("table")
            table_records = None if table is None else table.to_dict(orient="records")
            curve = result.get("the_curve") or {}
            serialized_result = {
                "R0": result.get("R0"),
                "S_H": result.get("S_H"),
                "R2": result.get("R2"),
                "offset": result.get("offset"),
                "table": table_records,
                "the_curve": {
                    "H": np.asarray(curve.get("H", [])).tolist(),
                    "Rxy_T": np.asarray(curve.get("Rxy_T", [])).tolist(),
                },
            }
        out[str(temp)] = {"params": params, "results": serialized_result}
    return out


def deserialize_session(data):
    """把 JSON 数据还原为 temp_data 结构。"""
    temp_data = {}
    for key, value in data.items():
        try:
            temp = float(key)
        except (TypeError, ValueError):
            temp = key

        params = value.get("params", dict(DEFAULT_PARAMS))
        result = value.get("results")
        results = None
        if result:
            table_records = result.get("table")
            table = None if table_records is None else pd.DataFrame(table_records)
            curve = result.get("the_curve") or {}
            results = {
                "R0": result.get("R0"),
                "S_H": result.get("S_H"),
                "R2": result.get("R2"),
                "offset": result.get("offset"),
                "table": table,
                "the_curve": {
                    "H": np.asarray(curve.get("H", []), dtype=float),
                    "Rxy_T": np.asarray(curve.get("Rxy_T", []), dtype=float),
                },
            }
        temp_data[temp] = {"params": params, "results": results}
    return temp_data


def _as_float(value, default):
    """安全地把值转成 float，失败时回退到默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _df_to_payload(df):
    """把 DataFrame 序列化为 JSON 字符串（orient=split），便于嵌套进会话 JSON。"""
    if df is None:
        return None
    return df.to_json(orient="split", force_ascii=False)


def _df_from_payload(text):
    """把 _df_to_payload 的输出还原为 DataFrame。"""
    if not text:
        return None
    return pd.read_json(io.StringIO(text), orient="split")


def build_session_payload(eto_df, vsm_df, width, thickness, length, vsm_min_fraction, temp_data):
    """构造完整会话 JSON：原始数据 + 全局参数 + 各温度 params/results。"""
    return {
        "version": 2,
        "sample": {
            "width": _as_float(width, SAMPLE_DEFAULTS["width"]),
            "thickness": _as_float(thickness, SAMPLE_DEFAULTS["thickness"]),
            "length": _as_float(length, SAMPLE_DEFAULTS["length"]),
        },
        "vsm_min_fraction": _as_float(vsm_min_fraction, 0.8),
        "eto": _df_to_payload(eto_df),
        "vsm": _df_to_payload(vsm_df),
        "temp_data": serialize_session(temp_data),
    }


def _clear_param_widget_keys():
    """清除各温度的控件 key，确保恢复后侧边栏重新从 params 初始化。"""
    prefixes = (
        "branch_",
        "remove_outliers_",
        "outlier_sigma_",
        "rxx_win_",
        "rxx_order_",
        "rxy_win_",
        "rxy_order_",
        "spline_s_",
        "zero_delta_",
        "use_h_range_",
        "hmin_",
        "hmax_",
        "use_fit_x_",
        "fitxmin_",
        "fitxmax_",
    )
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and any(key.startswith(p) for p in prefixes):
            del st.session_state[key]


def apply_restored_session(payload):
    """把恢复的会话写回 st.session_state，并准备恢复后的原始数据。"""
    if not isinstance(payload, dict):
        return

    sample = payload.get("sample") or {}
    _clear_param_widget_keys()
    st.session_state.temp_data = deserialize_session(payload.get("temp_data") or {})
    st.session_state["_restored_eto"] = _df_from_payload(payload.get("eto"))
    st.session_state["_restored_vsm"] = _df_from_payload(payload.get("vsm"))

    st.session_state[KEY_WIDTH] = _as_float(sample.get("width"), SAMPLE_DEFAULTS["width"])
    st.session_state[KEY_THICKNESS] = _as_float(sample.get("thickness"), SAMPLE_DEFAULTS["thickness"])
    st.session_state[KEY_LENGTH] = _as_float(sample.get("length"), SAMPLE_DEFAULTS["length"])
    st.session_state[KEY_VSM_FRACTION] = _as_float(payload.get("vsm_min_fraction"), 0.8)

    # 恢复后清空文件上传框，让解析区优先使用恢复的原始数据。
    st.session_state[KEY_ETO_UPLOADER] = None
    st.session_state[KEY_VSM_UPLOADER] = None


# --------------------------------------------------------------------------- #
# 页面入口
# --------------------------------------------------------------------------- #

def main() -> None:
    st.title("拓扑霍尔效应 (THE) 数据分析工具")

    if "temp_data" not in st.session_state:
        st.session_state.temp_data = {}
    if "show_summary" not in st.session_state:
        st.session_state.show_summary = False

    # ------------------------- 侧边栏：导入与样品尺寸 ------------------------- #
    with st.sidebar:
        st.header("恢复会话")
        restore_file = st.file_uploader(
            "恢复会话 JSON",
            type=["json"],
            key="session_restore",
            help="上传之前导出的 THE_session.json，恢复原始数据、样品参数与各温度分析结果。",
        )
        if restore_file is not None:
            raw = restore_file.getvalue()
            fingerprint = hashlib.md5(raw).hexdigest()
            if st.session_state.get("_restore_fingerprint") != fingerprint:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    apply_restored_session(payload)
                    st.session_state["_restore_fingerprint"] = fingerprint
                    st.success("会话已恢复。")
                except Exception as exc:
                    st.error(f"会话文件解析失败，请检查 JSON 格式：{exc}")

        st.divider()
        st.header("1. 数据导入")
        eto_file = st.file_uploader(
            "上传 ETO（输运）数据",
            type=["txt", "dat", "csv"],
            key="eto_uploader",
        )
        vsm_file = st.file_uploader(
            "上传 VSM（磁性）数据",
            type=["txt", "dat", "csv"],
            key="vsm_uploader",
        )

        st.divider()
        st.header("2. 样品尺寸")
        width = st.number_input(
            "宽度 W (μm)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.4f",
            help="用于电阻率换算 ρ = R·(W·t/L)，单位 μm。",
            key=KEY_WIDTH,
        )
        thickness = st.number_input(
            "厚度 t (μm)",
            min_value=0.0,
            value=0.1,
            step=0.01,
            format="%.4f",
            key=KEY_THICKNESS,
        )
        length = st.number_input(
            "电极间距 L (μm)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.4f",
            key=KEY_LENGTH,
        )

        if width > 0 and thickness > 0 and length > 0:
            # R 单位 Ω，W/t/L 单位 μm，ρ 单位 μΩ·cm：需乘 100。
            resistivity_factor = width * thickness / length * 100.0
            st.caption(
                f"电阻率换算因子 = {resistivity_factor:.6g}（μΩ·cm / Ω）"
            )
        else:
            resistivity_factor = None
            st.caption("电阻率换算需 W、t、L 均大于 0。")

        st.divider()
        st.header("VSM 温度筛选")
        vsm_min_fraction = st.slider(
            "保留数据点占比阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="仅保留 VSM 数据点数 ≥ 该比例 × 最大点数的温度。",
            key=KEY_VSM_FRACTION,
        )

    if resistivity_factor is None:
        resistivity_scale = 1.0
        st.warning("样品尺寸不完整，暂按原始电阻处理（未换算为电阻率）。")
    else:
        resistivity_scale = resistivity_factor

    # 磁场原始单位按 Oe 处理，换算到 T：1 Oe = 1e-4 T。
    field_scale = 1e-4

    # ------------------------- 解析上传文件 ------------------------- #
    eto_df = None
    vsm_df = None

    if eto_file is not None:
        with st.spinner("正在解析 ETO 数据…"):
            eto_df = parse_uploaded_file(eto_file.getvalue(), eto_file.name)
    elif st.session_state.get("_restored_eto") is not None:
        eto_df = st.session_state["_restored_eto"]

    if vsm_file is not None:
        with st.spinner("正在解析 VSM 数据…"):
            vsm_df = parse_uploaded_file(vsm_file.getvalue(), vsm_file.name)
    elif st.session_state.get("_restored_vsm") is not None:
        vsm_df = st.session_state["_restored_vsm"]

    if eto_df is None and vsm_df is None:
        st.info("请在左侧上传 ETO 和/或 VSM 数据文件，或恢复一个会话 JSON 后开始分析。")
        st.stop()

    # ------------------------- 按温度拆分与列识别 ------------------------- #
    eto_temp_col, eto_groups = split_by_temperature(eto_df)
    vsm_temp_col, vsm_groups = split_by_temperature(vsm_df)
    vsm_groups, vsm_counts = filter_temperatures_by_count(
        vsm_groups, vsm_min_fraction
    )

    eto_field_col = find_column(eto_df, FIELD_KEYWORDS)
    eto_rxx_col = find_column(eto_df, RXX_KEYWORDS)
    eto_rxy_col = find_column(eto_df, RXY_KEYWORDS)
    vsm_field_col = find_column(vsm_df, FIELD_KEYWORDS)
    vsm_moment_col = find_column(vsm_df, MOMENT_KEYWORDS)

    all_temperatures = sorted(set(eto_groups.keys()) | set(vsm_groups.keys()))

    # ------------------------- 侧边栏：温度与计算参数 ------------------------- #
    with st.sidebar:
        st.divider()
        st.header("3. 温度与扫描分支")
        if all_temperatures:
            selected_temperature = st.selectbox(
                "选择分析温度 (K)",
                all_temperatures,
                format_func=lambda value: f"{value:g} K",
            )
        else:
            selected_temperature = None
            st.warning("未能从数据中提取到温度，请检查文件格式或“温度”列。")

        if selected_temperature is not None:
            params = ensure_temp_params(selected_temperature)

            # 计算该温度下 ETO/VSM 的完整 H 范围，用于首次初始化 H 范围参数。
            h_values = []
            if eto_field_col and selected_temperature in eto_groups:
                h_values.append(
                    pd.to_numeric(
                        eto_groups[selected_temperature][eto_field_col],
                        errors="coerce",
                    ).dropna()
                )
            if vsm_field_col and selected_temperature in vsm_groups:
                h_values.append(
                    pd.to_numeric(
                        vsm_groups[selected_temperature][vsm_field_col],
                        errors="coerce",
                    ).dropna()
                )

            if h_values:
                all_h = pd.concat(h_values)
                full_h_min = float(all_h.min()) * field_scale
                full_h_max = float(all_h.max()) * field_scale
            else:
                full_h_min = full_h_max = 0.0

            if not params.get("h_range_initialized"):
                params["h_min"] = full_h_min
                params["h_max"] = full_h_max
                params["h_range_initialized"] = True

            branch_choice = st.selectbox(
                "场扫描分支",
                ["升场", "降场", "自动"],
                index=["升场", "降场", "自动"].index(params["branch"]),
                help="升场 = 磁场增大方向；降场 = 磁场减小方向。",
                key=f"branch_{selected_temperature}",
            )
            params["branch"] = branch_choice

            st.divider()
            st.header("4. 坏点去除")
            remove_outliers_enabled = st.checkbox(
                "启用坏点去除",
                value=params["remove_outliers"],
                key=f"remove_outliers_{selected_temperature}",
            )
            outlier_sigma = st.slider(
                "坏点阈值 (σ·MAD)",
                min_value=2.0,
                max_value=10.0,
                value=params["outlier_sigma"],
                step=0.5,
                key=f"outlier_sigma_{selected_temperature}",
            )
            params["remove_outliers"] = remove_outliers_enabled
            params["outlier_sigma"] = outlier_sigma

            st.divider()
            st.header("5. 平滑参数")
            st.markdown("**Rxx**")
            smooth_window_rxx = st.slider(
                "Rxx 窗口点数",
                min_value=3,
                max_value=101,
                value=params["rxx_window"],
                step=2,
                key=f"rxx_win_{selected_temperature}",
            )
            smooth_order_rxx = st.slider(
                "Rxx 多项式阶数",
                min_value=0,
                max_value=5,
                value=params["rxx_order"],
                step=1,
                key=f"rxx_order_{selected_temperature}",
            )
            params["rxx_window"] = smooth_window_rxx
            params["rxx_order"] = smooth_order_rxx

            st.markdown("**Rxy**")
            smooth_window_rxy = st.slider(
                "Rxy 窗口点数",
                min_value=3,
                max_value=101,
                value=params["rxy_window"],
                step=2,
                key=f"rxy_win_{selected_temperature}",
            )
            smooth_order_rxy = st.slider(
                "Rxy 多项式阶数",
                min_value=0,
                max_value=5,
                value=params["rxy_order"],
                step=1,
                key=f"rxy_order_{selected_temperature}",
            )
            params["rxy_window"] = smooth_window_rxy
            params["rxy_order"] = smooth_order_rxy

            st.markdown("**B样条插值（对称化 / 拟合）**")
            spline_s = st.number_input(
                "B样条平滑因子 S",
                min_value=0.0,
                value=params["spline_s"],
                step=1e-6,
                format="%.6g",
                key=f"spline_s_{selected_temperature}",
            )
            params["spline_s"] = spline_s

            st.divider()
            st.header("6. 零点校正")
            zero_delta = st.number_input(
                "零点偏移手动微调 Δ (μΩ·cm)",
                value=params["zero_delta"],
                step=1e-6,
                format="%.8g",
                key=f"zero_delta_{selected_temperature}",
            )
            params["zero_delta"] = zero_delta

            st.divider()
            st.header("7. 数据范围")
            use_h_range = st.checkbox(
                "手动选择数据范围",
                value=params["use_h_range"],
                key=f"use_h_range_{selected_temperature}",
            )
            params["use_h_range"] = use_h_range
            if use_h_range:
                col_hmin, col_hmax = st.columns(2)
                with col_hmin:
                    h_min = st.number_input(
                        "H 下限",
                        value=params["h_min"],
                        step=0.1,
                        format="%.6g",
                        key=f"hmin_{selected_temperature}",
                    )
                with col_hmax:
                    h_max = st.number_input(
                        "H 上限",
                        value=params["h_max"],
                        step=0.1,
                        format="%.6g",
                        key=f"hmax_{selected_temperature}",
                    )
                params["h_min"] = h_min
                params["h_max"] = h_max
                h_range = (h_min, h_max)
            else:
                h_range = None

            st.divider()
            st.header("8. 拟合范围")
            use_fit_x_range = st.checkbox(
                "手动指定拟合 X 范围",
                value=params["use_fit_x_range"],
                key=f"use_fit_x_{selected_temperature}",
            )
            params["use_fit_x_range"] = use_fit_x_range
            if use_fit_x_range:
                col_xmin, col_xmax = st.columns(2)
                with col_xmin:
                    fit_x_min = st.number_input(
                        "X 下限",
                        value=params["fit_x_min"],
                        step=1e-6,
                        format="%.6g",
                        key=f"fitxmin_{selected_temperature}",
                    )
                with col_xmax:
                    fit_x_max = st.number_input(
                        "X 上限",
                        value=params["fit_x_max"],
                        step=1e-6,
                        format="%.6g",
                        key=f"fitxmax_{selected_temperature}",
                    )
                params["fit_x_min"] = fit_x_min
                params["fit_x_max"] = fit_x_max

            st.divider()
            st.header("9. 生成汇总")
            if st.button("生成汇总", key="gen_summary"):
                st.session_state.show_summary = True

            # 将当前温度参数写回嵌套字典，保证跨页面/跨温度持久化。
            st.session_state.temp_data[selected_temperature]["params"] = params
        else:
            branch_choice = DEFAULT_PARAMS["branch"]
            smooth_window_rxx = DEFAULT_PARAMS["rxx_window"]
            smooth_order_rxx = DEFAULT_PARAMS["rxx_order"]
            smooth_window_rxy = DEFAULT_PARAMS["rxy_window"]
            smooth_order_rxy = DEFAULT_PARAMS["rxy_order"]
            spline_s = DEFAULT_PARAMS["spline_s"]
            zero_delta = DEFAULT_PARAMS["zero_delta"]
            remove_outliers_enabled = DEFAULT_PARAMS["remove_outliers"]
            outlier_sigma = DEFAULT_PARAMS["outlier_sigma"]
            use_h_range = DEFAULT_PARAMS["use_h_range"]
            h_range = None

        st.divider()
        st.header("10. 会话保存")
        session_json = json.dumps(
            build_session_payload(
                eto_df,
                vsm_df,
                width,
                thickness,
                length,
                vsm_min_fraction,
                st.session_state.temp_data,
            ),
            ensure_ascii=False,
            allow_nan=True,
        )
        st.download_button(
            "导出会话 JSON",
            session_json,
            file_name="THE_session.json",
            mime="application/json",
        )

    # ------------------------- 主区域：标签页 ------------------------- #
    tab_preview, tab_diagnostic, tab_table, tab_smooth, tab_sym, tab_fit = st.tabs(
        [
            "数据预览",
            "多温度诊断",
            "分温度数据表",
            "预处理与平滑",
            "零点校正与对称化",
            "标度拟合与 THE",
        ]
    )

    # ---- 数据预览 ----
    with tab_preview:
        col_eto, col_vsm = st.columns(2)
        with col_eto:
            if eto_df is not None:
                st.metric("ETO 数据行数", f"{len(eto_df):,}")
            else:
                st.metric("ETO 数据", "未上传")
        with col_vsm:
            if vsm_df is not None:
                st.metric("VSM 数据行数", f"{len(vsm_df):,}")
            else:
                st.metric("VSM 数据", "未上传")

        if all_temperatures:
            st.caption(
                f"共识别到 {len(all_temperatures)} 个温度点，范围 "
                f"{min(all_temperatures):g} – {max(all_temperatures):g} K。"
            )

        if vsm_df is not None and vsm_counts:
            max_count = max(vsm_counts.values())
            threshold = max(1, int(round(max_count * vsm_min_fraction)))
            st.caption(
                f"VSM 温度筛选：保留 {len(vsm_groups)}/{len(vsm_counts)} 个温度"
                f"（阈值 {vsm_min_fraction:.0%} × 最大点数 {max_count} = {threshold} 点）。"
            )

        if selected_temperature is not None:
            st.subheader(f"选中温度 {selected_temperature:g} K 的数据预览")
            preview_columns = st.columns(2)
            with preview_columns[0]:
                if eto_groups and selected_temperature in eto_groups:
                    st.markdown("**ETO（输运）数据 — 前 5 行**")
                    st.dataframe(eto_groups[selected_temperature].head(5))
                else:
                    st.info("该温度下没有 ETO 数据。")
            with preview_columns[1]:
                if vsm_groups and selected_temperature in vsm_groups:
                    st.markdown("**VSM（磁性）数据 — 前 5 行**")
                    st.dataframe(vsm_groups[selected_temperature].head(5))
                else:
                    st.info("该温度下没有 VSM 数据。")
        else:
            st.warning("没有可预览的温度数据。")

    # ---- 多温度诊断 ----
    with tab_diagnostic:
        if eto_df is None:
            st.info("上传 ETO 数据后，此处将叠加显示各温度的 Rxy(H)、Rxx(H) 与 VSM M(H)。")
        elif eto_field_col is None or eto_rxy_col is None:
            st.warning(
                "未能自动识别磁场列或 Rxy 列，跳过诊断图。"
                f"（已识别列：{', '.join(map(str, eto_df.columns))}）"
            )
        else:
            st.caption(
                f"通道映射：Rxx = {eto_rxx_col or '未识别'}，Rxy = {eto_rxy_col}。"
                "两个通道交替读数，绘图与统计时已去除各自通道的空值。"
            )

            fig_rxy = go.Figure()
            offset_rows = []
            for temp, group in sorted(eto_groups.items()):
                field = pd.to_numeric(group[eto_field_col], errors="coerce")
                rxy = pd.to_numeric(group[eto_rxy_col], errors="coerce")
                valid = field.notna() & rxy.notna()
                field = field[valid]
                rxy = rxy[valid]
                field = field * field_scale
                rxy = rxy * resistivity_scale
                if len(field) == 0:
                    continue

                order = np.argsort(field.to_numpy())
                fig_rxy.add_trace(
                    go.Scatter(
                        x=field.to_numpy()[order],
                        y=rxy.to_numpy()[order],
                        mode="lines+markers",
                        name=f"{temp:g} K",
                    )
                )

                offset = compute_zero_field_offset(field, rxy)
                if offset is not None:
                    offset_rows.append(
                        {
                            "温度 (K)": temp,
                            "零点偏移 ρxy(H≈0) (μΩ·cm)": offset,
                            "数据点数": int(len(field)),
                        }
                    )

            fig_rxy.update_layout(
                title="各温度 ρxy vs 磁场",
                xaxis_title="H (T)",
                yaxis_title="ρxy (μΩ·cm)",
                height=520,
                legend_title="温度",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig_rxy, use_container_width=True)

            if offset_rows:
                st.markdown("**各温度零点偏移统计**")
                st.dataframe(pd.DataFrame(offset_rows))
            else:
                st.info("没有可用于零点偏移统计的数据点。")

            if eto_rxx_col is not None:
                fig_rxx = go.Figure()
                for temp, group in sorted(eto_groups.items()):
                    field = pd.to_numeric(group[eto_field_col], errors="coerce")
                    rxx = pd.to_numeric(group[eto_rxx_col], errors="coerce")
                    valid = field.notna() & rxx.notna()
                    field = field[valid]
                    rxx = rxx[valid]
                    field = field * field_scale
                    rxx = rxx * resistivity_scale
                    if len(field) == 0:
                        continue
                    order = np.argsort(field.to_numpy())
                    fig_rxx.add_trace(
                        go.Scatter(
                            x=field.to_numpy()[order],
                            y=rxx.to_numpy()[order],
                            mode="lines+markers",
                            name=f"{temp:g} K",
                        )
                    )
                fig_rxx.update_layout(
                    title="各温度 ρxx vs 磁场",
                    xaxis_title="H (T)",
                    yaxis_title="ρxx (μΩ·cm)",
                    height=420,
                    legend_title="温度",
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(fig_rxx, use_container_width=True)
            else:
                st.info("未识别到 Rxx 列，跳过 Rxx 叠加图。")

            if (
                vsm_df is not None
                and vsm_field_col is not None
                and vsm_moment_col is not None
            ):
                fig_vsm = go.Figure()
                for temp, group in sorted(vsm_groups.items()):
                    field = pd.to_numeric(group[vsm_field_col], errors="coerce")
                    moment = pd.to_numeric(group[vsm_moment_col], errors="coerce")
                    valid = field.notna() & moment.notna()
                    field = field[valid]
                    moment = moment[valid]
                    field = field * field_scale
                    if len(field) == 0:
                        continue
                    order = np.argsort(field.to_numpy())
                    fig_vsm.add_trace(
                        go.Scatter(
                            x=field.to_numpy()[order],
                            y=moment.to_numpy()[order],
                            mode="lines+markers",
                            name=f"{temp:g} K",
                        )
                    )
                fig_vsm.update_layout(
                    title=f"各温度 M vs 磁场（VSM，列：{vsm_field_col} / {vsm_moment_col}）",
                    xaxis_title="H (T)",
                    yaxis_title=vsm_moment_col,
                    height=420,
                    legend_title="温度",
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(fig_vsm, use_container_width=True)
            else:
                st.info("未上传 VSM 或未识别到磁场/磁矩列，跳过 VSM 叠加图。")

    # ---- 分温度数据表 ----
    with tab_table:
        if selected_temperature is None:
            st.info("请先在侧边栏选择温度。")
        else:
            st.subheader(f"{selected_temperature:g} K 分温度数据表")

            eto_group = eto_groups.get(selected_temperature)
            vsm_group = vsm_groups.get(selected_temperature)

            rxx_series = extract_channel_series(
                eto_group,
                eto_field_col,
                eto_rxx_col,
                resistivity_scale,
                field_scale,
            )
            rxy_series = extract_channel_series(
                eto_group,
                eto_field_col,
                eto_rxy_col,
                resistivity_scale,
                field_scale,
            )
            vsm_series = extract_channel_series(
                vsm_group, vsm_field_col, vsm_moment_col, 1.0, field_scale
            )

            rxx_prep, rxx_label, _, rxx_removed = prepare_eto_series(
                rxx_series,
                branch_choice,
                h_range,
                remove_outliers_enabled,
                outlier_sigma,
                smooth_window_rxx,
                smooth_order_rxx,
            )
            rxy_prep, rxy_label, _, rxy_removed = prepare_eto_series(
                rxy_series,
                branch_choice,
                h_range,
                remove_outliers_enabled,
                outlier_sigma,
                smooth_window_rxy,
                smooth_order_rxy,
            )
            vsm_prep, vsm_removed = prepare_vsm_series(
                vsm_series,
                h_range,
                remove_outliers_enabled,
                outlier_sigma,
            )

            st.caption(
                f"分支：{rxy_label or rxx_label}；"
                f"范围：{'手动' if use_h_range else '全部'}；"
                f"坏点去除：{'开' if remove_outliers_enabled else '关'}。"
            )

            col_rxx, col_rxy, col_vsm = st.columns(3)
            with col_rxx:
                st.markdown(
                    f"**Rxx** — 剔除 {rxx_removed} 点，"
                    f"剩 {0 if rxx_prep is None else len(rxx_prep)} 点"
                )
                if rxx_prep is not None and not rxx_prep.empty:
                    st.dataframe(
                        rxx_prep[["H", "value"]]
                        .rename(columns={"value": "ρxx (μΩ·cm)"})
                        .reset_index(drop=True)
                    )
                else:
                    st.info("无数据。")

            with col_rxy:
                st.markdown(
                    f"**Rxy** — 剔除 {rxy_removed} 点，"
                    f"剩 {0 if rxy_prep is None else len(rxy_prep)} 点"
                )
                if rxy_prep is not None and not rxy_prep.empty:
                    st.dataframe(
                        rxy_prep[["H", "value"]]
                        .rename(columns={"value": "ρxy (μΩ·cm)"})
                        .reset_index(drop=True)
                    )
                else:
                    st.info("无数据。")

            with col_vsm:
                st.markdown(
                    f"**VSM** — 剔除 {vsm_removed} 点，"
                    f"剩 {0 if vsm_prep is None else len(vsm_prep)} 点"
                )
                if vsm_prep is not None and not vsm_prep.empty:
                    st.dataframe(
                        vsm_prep[["H", "value"]]
                        .rename(columns={"value": "M"})
                        .reset_index(drop=True)
                    )
                else:
                    st.info("无数据。")

            if st.button("确认使用当前数据范围", key="confirm_range"):
                st.session_state["range_confirmed"] = True
            if st.session_state.get("range_confirmed"):
                st.success("已确认：后续平滑 / 对称化将使用当前范围与坏点去除设置。")

    # ---- 预处理与平滑 ----
    with tab_smooth:
        if selected_temperature is None or eto_df is None:
            st.info("请上传 ETO 数据并在侧边栏选择温度。")
        elif eto_field_col is None or eto_rxx_col is None or eto_rxy_col is None:
            st.warning(
                "ETO 中未识别到磁场 / Rxx / Rxy 列，无法进行预处理。"
                f"（已识别列：{', '.join(map(str, eto_df.columns))}）"
            )
        else:
            group = eto_groups.get(selected_temperature)
            rxx_series = extract_channel_series(
                group, eto_field_col, eto_rxx_col, resistivity_scale, field_scale
            )
            rxy_series = extract_channel_series(
                group, eto_field_col, eto_rxy_col, resistivity_scale, field_scale
            )

            if rxx_series.empty and rxy_series.empty:
                st.warning("该温度下没有可用的输运数据。")
            else:
                rxx_branch, rxx_label, _, rxx_removed = prepare_eto_series(
                    rxx_series,
                    branch_choice,
                    h_range,
                    remove_outliers_enabled,
                    outlier_sigma,
                    smooth_window_rxx,
                    smooth_order_rxx,
                )
                rxy_branch, rxy_label, _, rxy_removed = prepare_eto_series(
                    rxy_series,
                    branch_choice,
                    h_range,
                    remove_outliers_enabled,
                    outlier_sigma,
                    smooth_window_rxy,
                    smooth_order_rxy,
                )
                st.caption(
                    f"分支：{rxx_label or rxy_label}；"
                    f"范围：{'手动' if use_h_range else '全部'}；"
                    f"坏点去除：{'开' if remove_outliers_enabled else '关'}；"
                    f"剔除 Rxx {rxx_removed} 点 / Rxy {rxy_removed} 点"
                )

                col_rxx, col_rxy = st.columns(2)
                with col_rxx:
                    if rxx_branch is not None and not rxx_branch.empty:
                        prep_rxx = smooth_series_df(
                            rxx_branch, smooth_window_rxx, smooth_order_rxx
                        )
                        fig_rxx = go.Figure()
                        fig_rxx.add_trace(
                            go.Scatter(
                                x=prep_rxx["H"],
                                y=prep_rxx["value"],
                                mode="markers",
                                name="ρxx 原始",
                            )
                        )
                        fig_rxx.add_trace(
                            go.Scatter(
                                x=prep_rxx["H"],
                                y=prep_rxx["smooth"],
                                mode="lines",
                                name="ρxx 平滑",
                            )
                        )
                        fig_rxx.update_layout(
                            title="ρxx 平滑对比",
                            xaxis_title="H (T)",
                            yaxis_title="ρxx (μΩ·cm)",
                            height=420,
                            margin=dict(l=20, r=20, t=60, b=20),
                        )
                        st.plotly_chart(fig_rxx, use_container_width=True)
                    else:
                        st.info("该分支没有 Rxx 数据。")

                with col_rxy:
                    if rxy_branch is not None and not rxy_branch.empty:
                        prep_rxy = smooth_series_df(
                            rxy_branch, smooth_window_rxy, smooth_order_rxy
                        )
                        fig_rxy = go.Figure()
                        fig_rxy.add_trace(
                            go.Scatter(
                                x=prep_rxy["H"],
                                y=prep_rxy["value"],
                                mode="markers",
                                name="ρxy 原始",
                            )
                        )
                        fig_rxy.add_trace(
                            go.Scatter(
                                x=prep_rxy["H"],
                                y=prep_rxy["smooth"],
                                mode="lines",
                                name="ρxy 平滑",
                            )
                        )
                        fig_rxy.update_layout(
                            title="ρxy 平滑对比",
                            xaxis_title="H (T)",
                            yaxis_title="ρxy (μΩ·cm)",
                            height=420,
                            margin=dict(l=20, r=20, t=60, b=20),
                        )
                        st.plotly_chart(fig_rxy, use_container_width=True)
                    else:
                        st.info("该分支没有 Rxy 数据。")

    # ---- 零点校正与对称化 ----
    with tab_sym:
        if selected_temperature is None or eto_df is None:
            st.info("请上传 ETO 数据并在侧边栏选择温度。")
        elif eto_field_col is None or eto_rxx_col is None or eto_rxy_col is None:
            st.warning("ETO 中未识别到磁场 / Rxx / Rxy 列，无法进行对称化。")
        else:
            group = eto_groups.get(selected_temperature)
            rxx_series = extract_channel_series(
                group, eto_field_col, eto_rxx_col, resistivity_scale, field_scale
            )
            rxy_series = extract_channel_series(
                group, eto_field_col, eto_rxy_col, resistivity_scale, field_scale
            )

            if rxx_series.empty and rxy_series.empty:
                st.warning("该温度下没有可用的输运数据。")
            else:
                rxx_branch, rxx_label, _, _ = prepare_eto_series(
                    rxx_series,
                    branch_choice,
                    h_range,
                    remove_outliers_enabled,
                    outlier_sigma,
                    smooth_window_rxx,
                    smooth_order_rxx,
                )
                rxy_branch, rxy_label, _, _ = prepare_eto_series(
                    rxy_series,
                    branch_choice,
                    h_range,
                    remove_outliers_enabled,
                    outlier_sigma,
                    smooth_window_rxy,
                    smooth_order_rxy,
                )

                auto_offset = 0.0
                if rxy_branch is not None and not rxy_branch.empty:
                    computed_offset = compute_zero_field_offset(
                        rxy_branch["H"], rxy_branch["value"]
                    )
                    if computed_offset is not None:
                        auto_offset = computed_offset
                total_offset = auto_offset + zero_delta

                st.caption(
                    f"分支：{rxy_label or rxx_label}；自动零点偏移 "
                    f"{auto_offset:.6g} μΩ·cm，总校正 {total_offset:.6g} μΩ·cm。"
                )

                if rxy_branch is not None and not rxy_branch.empty:
                    prep_rxy = smooth_series_df(
                        rxy_branch, smooth_window_rxy, smooth_order_rxy
                    )
                    prep_rxy["corr_smooth"] = prep_rxy["smooth"] - total_offset

                    fig_corr = go.Figure()
                    fig_corr.add_trace(
                        go.Scatter(
                            x=prep_rxy["H"],
                            y=prep_rxy["value"],
                            mode="markers",
                            name="ρxy 原始",
                        )
                    )
                    fig_corr.add_trace(
                        go.Scatter(
                            x=prep_rxy["H"],
                            y=prep_rxy["corr_smooth"],
                            mode="lines",
                            name="ρxy 校正后(平滑)",
                        )
                    )
                    fig_corr.update_layout(
                        title="ρxy 零点校正",
                        xaxis_title="H (T)",
                        yaxis_title="ρxy (μΩ·cm)",
                        height=420,
                        margin=dict(l=20, r=20, t=60, b=20),
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("该分支没有 Rxy 数据。")

                if branch_choice not in ("升场", "降场"):
                    st.info(
                        "对称化需要单一升场/降场分支，请在侧边栏选择“升场”或“降场”。"
                    )
                elif (
                    rxx_branch is None
                    or rxx_branch.empty
                    or rxy_branch is None
                    or rxy_branch.empty
                ):
                    st.warning("升场/降场分支缺少 Rxx 或 Rxy 数据，无法对称化。")
                else:
                    sym = symmetrize_channels(
                        rxx_branch,
                        rxy_branch,
                        smooth_window_rxx,
                        smooth_order_rxx,
                        smooth_window_rxy,
                        smooth_order_rxy,
                        total_offset,
                        spline_s,
                    )
                    if sym is None:
                        st.warning("所选分支缺少正负场数据，无法完成对称化。")
                    else:
                        col_sym_rxx, col_sym_rxy = st.columns(2)
                        with col_sym_rxx:
                            fig_sym_rxx = go.Figure()
                            fig_sym_rxx.add_trace(
                                go.Scatter(
                                    x=sym["H"],
                                    y=sym["Rxx_sym"],
                                    mode="lines",
                                    name="ρxx_sym",
                                )
                            )
                            fig_sym_rxx.update_layout(
                                title="对称化 ρxx_sym(H)",
                                xaxis_title="H (T)",
                                yaxis_title="ρxx_sym (μΩ·cm)",
                                height=420,
                                margin=dict(l=20, r=20, t=60, b=20),
                            )
                            st.plotly_chart(fig_sym_rxx, use_container_width=True)

                        with col_sym_rxy:
                            fig_sym_rxy = go.Figure()
                            fig_sym_rxy.add_trace(
                                go.Scatter(
                                    x=sym["H"],
                                    y=sym["Rxy_anti"],
                                    mode="lines",
                                    name="ρxy_anti",
                                )
                            )
                            fig_sym_rxy.update_layout(
                                title="反对称化 ρxy_anti(H)",
                                xaxis_title="H (T)",
                                yaxis_title="ρxy_anti (μΩ·cm)",
                                height=420,
                                margin=dict(l=20, r=20, t=60, b=20),
                            )
                            st.plotly_chart(fig_sym_rxy, use_container_width=True)


    # ---- 标度拟合与 THE ----
    with tab_fit:
        if selected_temperature is None or eto_df is None:
            st.info("请上传 ETO 数据并在侧边栏选择温度。")
        elif eto_field_col is None or eto_rxx_col is None or eto_rxy_col is None:
            st.warning("ETO 中未识别到磁场 / Rxx / Rxy 列，无法拟合。")
        elif vsm_df is None or vsm_field_col is None or vsm_moment_col is None:
            st.warning("需要上传 VSM 数据并识别到磁场/磁矩列才能拟合。")
        else:
            group = eto_groups.get(selected_temperature)
            rxx_series = extract_channel_series(
                group, eto_field_col, eto_rxx_col, resistivity_scale, field_scale
            )
            rxy_series = extract_channel_series(
                group, eto_field_col, eto_rxy_col, resistivity_scale, field_scale
            )

            if rxx_series.empty and rxy_series.empty:
                st.warning("该温度下没有 ETO 输运数据。")
            else:
                rxx_branch, rxx_label, _, _ = prepare_eto_series(
                    rxx_series,
                    branch_choice,
                    h_range,
                    remove_outliers_enabled,
                    outlier_sigma,
                    smooth_window_rxx,
                    smooth_order_rxx,
                )
                rxy_branch, rxy_label, _, _ = prepare_eto_series(
                    rxy_series,
                    branch_choice,
                    h_range,
                    remove_outliers_enabled,
                    outlier_sigma,
                    smooth_window_rxy,
                    smooth_order_rxy,
                )

                if branch_choice not in ("升场", "降场"):
                    st.info("请先在侧边栏选择“升场”或“降场”分支。")
                elif (
                    rxx_branch is None
                    or rxx_branch.empty
                    or rxy_branch is None
                    or rxy_branch.empty
                ):
                    st.warning("该分支缺少 Rxx 或 Rxy 数据。")
                else:
                    auto_offset = 0.0
                    if not rxy_branch.empty:
                        computed_offset = compute_zero_field_offset(
                            rxy_branch["H"], rxy_branch["value"]
                        )
                        if computed_offset is not None:
                            auto_offset = computed_offset
                    total_offset = auto_offset + zero_delta

                    sym = symmetrize_channels(
                        rxx_branch,
                        rxy_branch,
                        smooth_window_rxx,
                        smooth_order_rxx,
                        smooth_window_rxy,
                        smooth_order_rxy,
                        total_offset,
                        spline_s,
                    )
                    if sym is None:
                        st.warning("对称化失败，无法进行标度拟合。")
                    else:
                        vsm_group = vsm_groups.get(selected_temperature)
                        vsm_series = extract_channel_series(
                            vsm_group,
                            vsm_field_col,
                            vsm_moment_col,
                            1.0,
                            field_scale,
                        )
                        vsm_clean, _ = prepare_vsm_series(
                            vsm_series,
                            h_range,
                            remove_outliers_enabled,
                            outlier_sigma,
                        )
                        if vsm_clean is None or vsm_clean.empty:
                            st.warning(
                                "该温度没有 VSM 数据（可能已被温度筛选去除）。"
                            )
                        else:
                            h_common = min(
                                float(sym["H"].max()), float(vsm_clean["H"].max())
                            )
                            grid = sym.loc[
                                sym["H"] <= h_common, "H"
                            ].to_numpy(dtype=float)
                            if len(grid) < 10:
                                st.warning("有效拟合磁场区间过短。")
                            else:
                                moment = interpolate_vsm_to_grid(
                                    vsm_clean, grid, spline_s
                                )
                                if moment is None:
                                    st.warning("VSM 磁矩插值失败。")
                                else:
                                    rxx_sym = np.interp(
                                        grid,
                                        sym["H"].to_numpy(dtype=float),
                                        sym["Rxx_sym"].to_numpy(dtype=float),
                                    )
                                    rxy_anti = np.interp(
                                        grid,
                                        sym["H"].to_numpy(dtype=float),
                                        sym["Rxy_anti"].to_numpy(dtype=float),
                                    )
                                    table = pd.DataFrame(
                                        {
                                            "H": grid,
                                            "Rxx_sym": rxx_sym,
                                            "Rxy_anti": rxy_anti,
                                            "M": moment,
                                        }
                                    )
                                    table = table[table["H"] > 1e-12]
                                    table["X"] = (
                                        table["Rxx_sym"] ** 2
                                        * table["M"]
                                        / table["H"]
                                    )
                                    table["Y"] = table["Rxy_anti"] / table["H"]
                                    table = table.replace(
                                        [np.inf, -np.inf], np.nan
                                    ).dropna()

                                    if len(table) < 5:
                                        st.warning("有效拟合点数不足。")
                                    else:
                                        table = table.reset_index(drop=True)
                                        fig_scale = go.Figure(
                                            go.Scatter(
                                                x=table["X"],
                                                y=table["Y"],
                                                mode="markers",
                                                name="标度数据",
                                                customdata=table["H"].to_numpy(
                                                    dtype=float
                                                ),
                                                hovertemplate=(
                                                    "X=%{x}<br>Y=%{y}<br>"
                                                    "H=%{customdata}<extra></extra>"
                                                ),
                                            )
                                        )
                                        fig_scale.update_layout(
                                            title="标度图 Y = ρxy_anti/H vs X = ρxx_sym²·M/H",
                                            xaxis_title="X = ρxx_sym²·M/H",
                                            yaxis_title="Y = ρxy_anti/H",
                                            height=420,
                                            margin=dict(l=20, r=20, t=60, b=20),
                                        )
                                        event = st.plotly_chart(
                                            fig_scale,
                                            key="scaling_select",
                                            on_select="rerun",
                                            selection_mode=(
                                                "points",
                                                "box",
                                                "lasso",
                                            ),
                                            use_container_width=True,
                                        )

                                        points = []
                                        if event is not None:
                                            selection = getattr(
                                                event, "selection", None
                                            )
                                            if selection is not None:
                                                points = selection.points or []

                                        selected_h = []
                                        point_indices = []
                                        for point in points:
                                            customdata = point.get("customdata")
                                            if isinstance(customdata, (list, tuple)):
                                                if len(customdata) > 0:
                                                    selected_h.append(customdata[0])
                                            elif customdata is not None:
                                                selected_h.append(customdata)
                                            index = point.get("point_index")
                                            if index is not None:
                                                point_indices.append(int(index))

                                        if selected_h:
                                            selected = table[
                                                table["H"].isin(selected_h)
                                            ].copy()
                                        elif point_indices:
                                            selected = table.iloc[
                                                point_indices
                                            ].copy()
                                        else:
                                            selected = table.iloc[:0].copy()

                                        if selected.empty:
                                            st.info(
                                                "请在标度图中用框选或套索选择线性区；"
                                                "选中后下方会显示各点对应的 H 并自动拟合。"
                                            )
                                        elif len(selected) < 3:
                                            st.warning("请至少框选 3 个点。")
                                        else:
                                            st.subheader(
                                                f"已选中 {len(selected)} 个点（含对应 H）"
                                            )
                                            st.dataframe(
                                                selected[
                                                    [
                                                        "H",
                                                        "X",
                                                        "Y",
                                                        "Rxx_sym",
                                                        "Rxy_anti",
                                                        "M",
                                                    ]
                                                ]
                                                .rename(
                                                    columns={
                                                        "Rxx_sym": "ρxx_sym (μΩ·cm)",
                                                        "Rxy_anti": "ρxy_anti (μΩ·cm)",
                                                    }
                                                )
                                                .reset_index(drop=True)
                                            )

                                            regression = linregress(
                                                selected["X"], selected["Y"]
                                            )
                                            r0 = float(regression.intercept)
                                            s_h = float(regression.slope)
                                            r2 = float(regression.rvalue ** 2)

                                            col_r0, col_sh, col_r2 = st.columns(3)
                                            col_r0.metric("R₀ (μΩ·cm/T)", f"{r0:.6g}")
                                            col_sh.metric("S_H", f"{s_h:.6g}")
                                            col_r2.metric("R²", f"{r2:.6g}")

                                            table["Rxy_T"] = (
                                                table["Rxy_anti"]
                                                - r0 * table["H"]
                                                - s_h
                                                * (table["Rxx_sym"] ** 2)
                                                * table["M"]
                                            )

                                            fig_the = go.Figure(
                                                go.Scatter(
                                                    x=table["H"],
                                                    y=table["Rxy_T"],
                                                    mode="lines",
                                                    name="ρxy_T",
                                                )
                                            )
                                            fig_the.update_layout(
                                                title="拓扑霍尔电阻率 ρxy_T(H)",
                                                xaxis_title="H (T)",
                                                yaxis_title="ρxy_T (μΩ·cm)",
                                                height=420,
                                                margin=dict(l=20, r=20, t=60, b=20),
                                            )
                                            st.plotly_chart(
                                                fig_the, use_container_width=True
                                            )

                                            st.download_button(
                                                "下载拟合结果 CSV",
                                                table.to_csv(index=False).encode(
                                                    "utf-8-sig"
                                                ),
                                                file_name=(
                                                    f"THE_fit_{selected_temperature:g}K.csv"
                                                ),
                                                mime="text/csv",
                                            )

                                            st.divider()
                                            if st.button(
                                                "保存该温度拟合结果",
                                                key=f"save_fit_{selected_temperature}",
                                            ):
                                                ensure_temp_params(selected_temperature)
                                                data = st.session_state.temp_data
                                                data[selected_temperature][
                                                    "results"
                                                ] = {
                                                    "R0": r0,
                                                    "S_H": s_h,
                                                    "R2": r2,
                                                    "offset": total_offset,
                                                    "table": table.copy(),
                                                    "the_curve": {
                                                        "H": table["H"].to_numpy(
                                                            dtype=float
                                                        ),
                                                        "Rxy_T": table[
                                                            "Rxy_T"
                                                        ].to_numpy(dtype=float),
                                                    },
                                                }
                                                params_saved = data[selected_temperature][
                                                    "params"
                                                ]
                                                params_saved["use_fit_x_range"] = True
                                                params_saved["fit_x_min"] = float(
                                                    selected["X"].min()
                                                )
                                                params_saved["fit_x_max"] = float(
                                                    selected["X"].max()
                                                )
                                                data[selected_temperature][
                                                    "params"
                                                ] = params_saved
                                                st.success(
                                                    "该温度拟合结果已保存，汇总时会收集。"
                                                )


    # ------------------------- 多温度汇总 ------------------------- #
    if st.session_state.get("show_summary"):
        st.divider()
        st.subheader("多温度分析汇总")

        analyzed = []
        for temp in sorted(st.session_state.temp_data.keys()):
            entry = st.session_state.temp_data[temp]
            result = entry.get("results")
            if result:
                analyzed.append((temp, entry.get("params", {}), result))

        if not analyzed:
            st.info(
                "还没有已分析的温度：可在“标度拟合与 THE”页框选线性区后"
                "点击“保存该温度拟合结果”。"
            )
        else:
            rows = []
            for temp, params, result in analyzed:
                rows.append(
                    {
                        "温度 (K)": temp,
                        "分支": params.get("branch"),
                        "Rxx窗口": params.get("rxx_window"),
                        "Rxx阶数": params.get("rxx_order"),
                        "Rxy窗口": params.get("rxy_window"),
                        "Rxy阶数": params.get("rxy_order"),
                        "B样条S": params.get("spline_s"),
                        "零点Δ (μΩ·cm)": params.get("zero_delta"),
                        "R0 (μΩ·cm/T)": result.get("R0"),
                        "S_H": result.get("S_H"),
                        "R²": result.get("R2"),
                    }
                )

            st.markdown("**参数与结果汇总表**")
            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df)
            st.download_button(
                "下载参数汇总表 CSV",
                summary_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="THE_summary.csv",
                mime="text/csv",
            )

            fig_overlay = go.Figure()
            for temp, _, result in sorted(analyzed):
                curve = result.get("the_curve") or {}
                h = curve.get("H")
                rxy_t = curve.get("Rxy_T")
                if h is not None and rxy_t is not None:
                    fig_overlay.add_trace(
                        go.Scatter(
                            x=h,
                            y=rxy_t,
                            mode="lines",
                            name=f"{temp:g} K",
                        )
                    )

            fig_overlay.update_layout(
                title="各温度 ρxy_T(H) 叠加",
                xaxis_title="H (T)",
                yaxis_title="ρxy_T (μΩ·cm)",
                height=520,
                legend_title="温度",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig_overlay, use_container_width=True)

            curve_rows = []
            for temp, _, result in sorted(analyzed):
                curve = result.get("the_curve") or {}
                h = curve.get("H")
                rxy_t = curve.get("Rxy_T")
                if h is not None and rxy_t is not None:
                    for hi, yi in zip(h, rxy_t):
                        curve_rows.append(
                            {
                                "温度 (K)": temp,
                                "H (T)": hi,
                                "ρxy_T (μΩ·cm)": yi,
                            }
                        )

            if curve_rows:
                curves_df = pd.DataFrame(curve_rows)
                st.download_button(
                    "下载 THE 曲线数据 CSV",
                    curves_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="THE_curves.csv",
                    mime="text/csv",
                )

            if st.button("关闭汇总", key="close_summary"):
                st.session_state.show_summary = False
                st.rerun()


if __name__ == "__main__":
    main()
