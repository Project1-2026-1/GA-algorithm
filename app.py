"""
GIAO DIỆN CHÍNH (Streamlit). Chủ sở hữu: TV3.
Chạy: streamlit run app.py
Chỉ gọi src.core.api (không import thẳng parser/GA), để tích hợp không vỡ khi module bên dưới thay đổi.
"""
from __future__ import annotations

import os
import sys

# Đảm bảo đường dẫn import cho src và root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.core import api
from src.core.types import GAConfig, ParseError, SolveRequest

st.set_page_config(page_title="GA Giải Phương Trình", page_icon="📐", layout="wide")

# Khai báo component MathLive.io
_CANDIDATE_PATHS = [
    os.path.join(ROOT_DIR, "src", "ga_solver", "frontend"),
    os.path.join(ROOT_DIR, "src", "frontend", "mathlive"),
    os.path.join(ROOT_DIR, "src", "frontend"),
]
_MATHLIVE_PATH = next((p for p in _CANDIDATE_PATHS if os.path.isfile(os.path.join(p, "index.html"))), _CANDIDATE_PATHS[0])
try:
    mathlive_field = components.declare_component("mathlive_field", path=_MATHLIVE_PATH)
except Exception:
    mathlive_field = None


# ----------------- SIDEBAR CẤU HÌNH THUẬT TOÁN GA -----------------
with st.sidebar:
    st.header("⚙️ Tham số GA")
    cfg = GAConfig(
        pop_size=st.slider("Kích thước quần thể (Pop Size)", 20, 500, 100, 10),
        generations=st.slider("Số thế hệ (Generations)", 20, 2000, 300, 20),
        crossover_rate=st.slider("Xác suất lai ghép (Crossover)", 0.0, 1.0, 0.9, 0.05),
        mutation_rate=st.slider("Xác suất đột biến (Mutation)", 0.0, 1.0, 0.1, 0.01),
        seed=int(st.number_input("Seed ngẫu nhiên", value=42, step=1)),
        find_all_roots=st.checkbox("Tìm nhiều nghiệm (Niching)", value=True),
    )
    st.markdown("---")
    run_baseline = st.checkbox("So sánh với phương pháp cổ điển (Baseline)", value=True)


# ----------------- THANH MENU ĐIỀU HƯỚNG TRÊN CÙNG -----------------
tab_solve, tab_guide = st.tabs(["🧮 Giải Phương Trình", "📖 Hướng Dẫn Sử Dụng"])


# ===================================================================
# TAB 1: GIẢI PHƯƠNG TRÌNH
# ===================================================================
with tab_solve:
    st.title("Giải Phương Trình Bằng Giải Thuật Di Truyền")
    st.caption("Ứng dụng giải phương trình và hệ phương trình phi tuyến số thực kết hợp so sánh với thuật toán cổ điển.")

    # Khởi tạo session state cho danh sách phương trình
    if "equations_list" not in st.session_state:
        st.session_state["equations_list"] = ["x + 9 = 5"]

    st.markdown("##### 📝 Phương trình / Hệ phương trình cần giải:")

    # Khung nhập toán học đa dòng GeoGebra-style với MathLive
    if mathlive_field is not None:
        ml_val = mathlive_field(
            equations=st.session_state["equations_list"],
            key="mathlive_geogebra_editor"
        )
        if ml_val is not None:
            if isinstance(ml_val, list):
                parsed_eqs = [str(e).strip() for e in ml_val if str(e).strip()]
            else:
                parsed_eqs = [line.strip() for line in str(ml_val).replace(r"\\", "\n").splitlines() if line.strip()]

            if parsed_eqs:
                st.session_state["equations_list"] = parsed_eqs
    else:
        st.warning("⚠️ Đang chạy chế độ nhập văn bản dự phòng:")
        eq_text = st.text_area(
            "Nhập phương trình (mỗi phương trình một dòng):",
            value="\n".join(st.session_state["equations_list"]),
            key="fallback_equations"
        )
        st.session_state["equations_list"] = [line.strip() for line in eq_text.splitlines() if line.strip()]

    equations = [e for e in st.session_state.get("equations_list", []) if e.strip()]

    # ----------------- PHÂN TÍCH PHƯƠNG TRÌNH & KHOẢNG TÌM KIẾM -----------------
    if not equations:
        st.info("💡 Vui lòng nhập phương trình vào khung bên trên để bắt đầu.")
    else:
        try:
            variables, warns = api.detect_variables(equations)
            for w in warns:
                st.warning(w)
        except ParseError as e:
            st.error(f"Lỗi cú pháp phương trình: {e}")
            variables = []

        if variables:
            # # ----------------- HỘP KIỂM TRA DỮ LIỆU NẠP VÀO CORE -----------------
            with st.expander("🔍 Chi tiết dữ liệu nạp vào Core (Problem Inspector)", expanded=False):
                try:
                    prob = api._parser.parse_problem(equations)
                    st.markdown(f"- **Danh sách phương trình nhận được ({len(equations)} phương trình):**")
                    for i, eq in enumerate(equations, 1):
                        st.markdown(f"  - `({i}) {eq}`")
                    st.markdown(f"- **Danh sách biến (`Problem.variables`):** `{prob.variables}`")
                    st.markdown(f"- **Công thức chuẩn hoá (`Problem.equations`):** `{prob.equations}`")
                    st.markdown(f"- **Số phương trình (`Problem.n_eqs`):** `{prob.n_eqs}`")

                    # Thử nghiệm tính toán sai số tại điểm gốc X = 0
                    x_test = np.zeros((1, len(prob.variables)))
                    res_val = prob.residuals(x_test)[0]
                    obj_val = prob.objective(x_test)[0]
                    st.markdown(f"- **Thử nghiệm tính tại X = 0:** Phần dư $R(0) =$ `{res_val}`, Objective = `{obj_val:.4e}`")
                    st.success("✅ Parser đã chuyển đổi thành công phương trình thành hàm số NumPy vector hoá cho Core!")
                except Exception as ex:
                    st.error(f"Lỗi khi chuyển đổi: {ex}")

            st.markdown("##### 📍 Khoảng tìm kiếm (Bounds):")
            bounds = {}
            cols = st.columns(max(len(variables), 1))
            for col, v in zip(cols, variables):
                with col:
                    lo = st.number_input(f"{v} min", value=-10.0, key=f"lo_{v}")
                    hi = st.number_input(f"{v} max", value=10.0, key=f"hi_{v}")
                    bounds[v] = (lo, hi)

            # ----------------- NÚT GIẢI & HIỂN THỊ KẾT QUẢ -----------------
            if st.button("🚀 Bắt đầu giải (Run GA)", key="btn_solve", type="primary", use_container_width=True):
                with st.spinner("Đang chạy thuật toán di truyền tiến hoá tìm nghiệm..."):
                    resp = api.solve(SolveRequest(
                        equations=equations,
                        bounds=bounds,
                        config=cfg,
                        run_baseline=run_baseline
                    ))

                for w in resp.warnings:
                    st.warning(w)

                if not resp.ok:
                    st.error(resp.error)
                else:
                    p, ga = resp.problem, resp.ga
                    st.subheader(f"Kết quả GA: Tìm được {len(ga.roots)} nghiệm")

                    rows = [{**{v: float(r[i]) for i, v in enumerate(p.variables)},
                             "max|f|": p.max_abs_residual(r)} for r in ga.roots]
                    st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame(columns=p.variables + ["max|f|"]),
                                 use_container_width=True)
                    st.caption(f"Tổng số lần đánh giá hàm: {ga.n_evaluations} · Thời gian: {ga.elapsed_s:.3f} s · " + "; ".join(ga.notes))

                    # Vẽ đồ thị
                    fig_warns: list[str] = []
                    B = api.bounds_array(p, bounds)
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_conv = api.figure_convergence(resp, fig_warns)
                        if fig_conv is not None:
                            st.plotly_chart(fig_conv, use_container_width=True)
                    with c2:
                        fig_prob = api.figure_problem(resp, B, fig_warns)
                        if fig_prob is not None:
                            st.plotly_chart(fig_prob, use_container_width=True)
                    for w in fig_warns:
                        st.warning(w)

                    # Bảng so sánh với Baseline
                    if resp.baseline is not None:
                        b = resp.baseline
                        st.subheader("📊 Bảng so sánh với phương pháp cổ điển")
                        st.table(pd.DataFrame([
                            {"Phương pháp": "Giải thuật Di truyền (GA)", "Số nghiệm tìm được": len(ga.roots), "Thời gian (s)": round(ga.elapsed_s, 4)},
                            {"Phương pháp": b.method, "Số nghiệm tìm được": len(b.roots), "Thời gian (s)": round(b.elapsed_s, 4)},
                        ]))


# ===================================================================
# TAB 2: HƯỚNG DẪN SỬ DỤNG
# ===================================================================
with tab_guide:
    st.header("📖 Hướng Dẫn Sử Dụng & Phạm Vi Bài Toán")

    st.markdown("""
    Ứng dụng **Giải phương trình bằng Giải thuật Di truyền (GA)** giải quyết các bài toán tìm nghiệm số thực bằng cách chuyển đổi phương trình hoặc hệ phương trình về bài toán tối ưu hóa cực tiểu sai số:
    $$\\min \\text{objective}(X) = \\sum_{i=1}^m \\big(f_i(X)\\big)^2$$
    Nghiệm được xác định khi sai số phần dư $\\max |f_i(X)| < \\text{tol}$ (mặc định $10^{-6}$).
    """)

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("1. Hướng dẫn nhập & thao tác")
        st.markdown("""
        - **Bàn phím ảo MathLive (MathLive.io)**:
          - Chạm hoặc click chuột vào ô công thức để **tự động mở bàn phím ảo MathLive** ngay trên màn hình.
          - Có nút **"⌨️ Mở / Đóng Bàn Phím"** để chủ động bật/tắt khi cần.
          - Hỗ trợ đầy đủ các phím toán học trực quan: phân số $\\frac{a}{b}$, căn bậc hai $\\sqrt{x}$, số mũ $x^2$, lượng giác, hằng số $\\pi$.
        - **Cú pháp phương trình**:
          - Hỗ trợ cả công thức trực quan MathLive và cú pháp văn bản chuẩn: `x^3 - 2*x + sin(x) = 0`.
          - Có thể viết dạng đẳng thức hai vế: `x^2 + y^2 = 4` hoặc viết biểu thức: `x^2 - 2` (hệ thống tự hiểu là `= 0`).
          - Đối với hệ phương trình: Mở mục *'Nhập hệ phương trình'* để viết mỗi phương trình trên một dòng.
        - **Cài đặt khoảng tìm kiếm (Bounds)**:
          - Sau khi nhập phương trình, hệ thống tự động bóc tách các biến ẩn ($x, y, z...$).
          - Bạn cần thiết lập khoảng `[min, max]` hợp lý bao trọn vùng nghi ngờ có nghiệm để thuật toán hội tụ nhanh chóng.
        - **Cấu hình tham số GA ở thanh bên trái**:
          - *Kích thước quần thể (Pop Size)*: Càng lớn thì khám phá không gian càng rộng, nhưng tính toán chậm hơn.
          - *Số thế hệ (Generations)*: Số chu kỳ tiến hoá lặp lại.
          - *Tìm nhiều nghiệm (Niching)*: Bật chế độ tìm phân tán nhiều nghiệm trong không gian.
        """)

    with col_g2:
        st.subheader("2. Các loại phương trình ứng dụng giải được")
        st.markdown("""
        Nhờ cơ chế tìm kiếm toàn cục không phụ thuộc vào đạo hàm, GA đặc biệt hiệu quả trên các lớp bài toán:

        - 🌊 **Phương trình lượng giác (Trigonometric Equations)**:
          - Các hàm chứa $\\sin, \\cos, \\tan$ tuần hoàn có rất nhiều cực tiểu địa phương mà các phương pháp giải tích hoặc Newton-Raphson dễ bị mắc kẹt.
          - *Ví dụ*: `sin(x) = x/5`, `cos(x) * sin(2*x) = 0.5`.

        - 📈 **Phương trình Logarit & Hàm số mũ lớn (Exponential & Log Equations)**:
          - Phương trình siêu việt không có nghiệm giải tích dạng đóng, có độ dốc tăng trưởng cực nhanh khiến phương pháp giải tích truyền thống bị tràn số hoặc khó tính đạo hàm.
          - *Ví dụ*: `exp(x) - 5*x^2 = 0`, `log(x^2 + 1) + 2^x = 10`.

        - 🔄 **Phương trình Vi tích phân & Dạng xấp xỉ rời rạc (Differential/Integral Approximations)**:
          - Các bài toán tích phân hoặc vi phân phi tuyến sau khi được rời rạc hoá bằng công thức số trị (như công thức hình thang/Simpson hoặc sai phân hữu hạn Euler/Runge-Kutta).
          - GA giải quyết tìm giá trị tham số trạng thái thỏa mãn cân bằng sai phân một cách trực tiếp.

        - 🔢 **Đa thức bậc cao & Phương trình phi tuyến phức tạp**:
          - Các phương trình đa thức bậc $\\ge 5$ không tồn tại công thức nghiệm căn thức tổng quát (định lý Abel-Ruffini).
          - *Ví dụ*: `x^5 - 3*x^4 + 2*x^3 - x + 1 = 0`.

        - 🌐 **Hệ phương trình phi tuyến nhiều biến (Nonlinear Systems)**:
          - Tìm giao điểm giữa các mặt cong phi tuyến trong không gian nhiều chiều.
          - *Ví dụ hệ*:
            $$\\begin{cases} x^2 + y^2 = 4 \\\\ x - y = 1 \\end{cases}$$
        """)
