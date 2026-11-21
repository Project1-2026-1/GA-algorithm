from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
import numpy as np
import plotly.graph_objects as go

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from ga_solver.baselines import BaselineResult, solve_baseline
    from ga_solver.config import GAConfig, SolverConfig
    from ga_solver.ga.algorithm import run_ga
    from ga_solver.parser import expression as _parser
    from ga_solver.parser.expression import ParseError, Problem, detect_variables, parse_problem
    from ga_solver.solver import find_roots
except ImportError:
    from src.ga_solver.baselines import BaselineResult, solve_baseline
    from src.ga_solver.config import GAConfig, SolverConfig
    from src.ga_solver.ga.algorithm import run_ga
    from src.ga_solver.parser import expression as _parser
    from src.ga_solver.parser.expression import ParseError, Problem, detect_variables, parse_problem
    from src.ga_solver.solver import find_roots

from src.core.types import GARunResult, SolveRequest, SolveResponse


def bounds_array(problem: Problem, bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    """Chuyển đổi dictionary bounds thành mảng NumPy shape (n_vars, 2)."""
    rows = []
    for v in problem.variables:
        b = bounds.get(v, (-10.0, 10.0))
        rows.append([float(b[0]), float(b[1])])
    return np.array(rows, dtype=float)


def solve(req: SolveRequest) -> SolveResponse:
    """Giải bài toán bằng GA và phương pháp cổ điển."""
    warnings: list[str] = []
    try:
        prob = parse_problem(req.equations)
    except ParseError as e:
        return SolveResponse(ok=False, error=str(e), warnings=[])
    except Exception as e:
        return SolveResponse(ok=False, error=f"Lỗi không xác định khi phân tích phương trình: {e}", warnings=[])

    n_vars = len(prob.variables)
    B = bounds_array(prob, req.bounds)

    start_time = time.perf_counter()
    ga_roots: list[np.ndarray] = []
    hist_best: list[float] = []
    hist_mean: list[float] = []
    n_evals = 0
    notes: list[str] = []

    if n_vars == 1:
        v_name = prob.variables[0]
        lo, hi = float(B[0, 0]), float(B[0, 1])
        cfg = replace(req.config, lower_bound=lo, upper_bound=hi)
        sc = SolverConfig()

        def f_1d(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
            res = prob.residuals(x_arr)
            if prob.n_eqs == 1:
                return res[:, 0]
            return np.sqrt(np.sum(res**2, axis=1))

        if getattr(req.config, "find_all_roots", True):
            solve_res = find_roots(f_1d, ga_config=cfg, solver_config=sc)
            for r in solve_res.roots:
                ga_roots.append(np.array([r.x]))
            n_evals = sum(len(r.best_history) * cfg.pop_size for r in solve_res.runs)
            if solve_res.runs:
                hist_best = solve_res.runs[0].best_history
                hist_mean = solve_res.runs[0].mean_history
            notes.append(f"Đã duyệt {len(solve_res.runs)} chu kỳ GA chia khoảng con")
        else:
            ga_res = run_ga(f_1d, cfg)
            if ga_res.best_error < 1e-4:
                ga_roots.append(np.array([ga_res.best_x]))
            n_evals = len(ga_res.best_history) * cfg.pop_size
            hist_best = ga_res.best_history
            hist_mean = ga_res.mean_history
            notes.append(f"Chạy GA đơn lẻ qua {ga_res.n_generations} thế hệ")

    else:
        # Giải hệ nhiều biến bằng GA đa biến kết hợp multi-start niching
        rng = np.random.default_rng(req.config.seed or 42)
        n_restarts = 10 if getattr(req.config, "find_all_roots", True) else 1
        pop_size = req.config.pop_size
        gens = req.config.generations

        for r_idx in range(n_restarts):
            # Khởi tạo quần thể ngẫu nhiên trong miền [lo, hi]
            pop = rng.uniform(B[:, 0], B[:, 1], size=(pop_size, n_vars))
            stall = 0
            best_err_run = np.inf

            for g in range(gens):
                err = np.sqrt(prob.objective(pop))
                n_evals += pop_size
                fit = 1.0 / (1.0 + err)

                min_err = float(np.min(err))
                if r_idx == 0:
                    hist_best.append(float(np.max(fit)))
                    hist_mean.append(float(np.mean(fit)))

                if min_err < 1e-8:
                    break
                if min_err < best_err_run * (1 - 1e-9):
                    best_err_run = min_err
                    stall = 0
                else:
                    stall += 1
                    if stall >= 40:
                        break

                # Elitism
                elite_idx = np.argsort(err)[:req.config.elite_size]
                elites = pop[elite_idx].copy()

                # Tournament selection
                n_children = pop_size - req.config.elite_size
                n_parents = n_children + (n_children % 2)
                k = min(req.config.tournament_k, pop_size)
                cand = np.argsort(rng.random((n_parents, pop_size)), axis=1)[:, :k]
                winners = cand[np.arange(n_parents), np.argmax(fit[cand], axis=1)]
                parents = pop[winners]
                rng.shuffle(parents)

                # Arithmetic Crossover
                children = np.empty((n_parents, n_vars))
                for i in range(0, n_parents, 2):
                    if rng.random() < req.config.crossover_rate:
                        alpha = rng.random()
                        children[i] = alpha * parents[i] + (1 - alpha) * parents[i + 1]
                        children[i + 1] = alpha * parents[i + 1] + (1 - alpha) * parents[i]
                    else:
                        children[i] = parents[i].copy()
                        children[i + 1] = parents[i + 1].copy()

                # Mutation
                child_sub = children[:n_children].copy()
                mask = rng.random(child_sub.shape) < req.config.mutation_rate
                scale = 0.02 * (B[:, 1] - B[:, 0])
                noise = rng.normal(0, scale, child_sub.shape)
                child_sub += mask * noise
                child_sub = np.clip(child_sub, B[:, 0], B[:, 1])

                pop = np.vstack([elites, child_sub])

            err_final = np.sqrt(prob.objective(pop))
            best_idx = np.argmin(err_final)
            cand_x = pop[best_idx].copy()
            if err_final[best_idx] < 1e-2:
                # Tinh chỉnh cục bộ bằng Scipy L-BFGS-B nếu khả thi
                try:
                    from scipy.optimize import minimize
                    res_opt = minimize(
                        lambda pt: prob.objective(pt.reshape(1, -1))[0],
                        cand_x,
                        bounds=B,
                        method="L-BFGS-B",
                    )
                    if res_opt.fun < prob.objective(cand_x.reshape(1, -1))[0]:
                        cand_x = res_opt.x
                except Exception:
                    pass

            if prob.max_abs_residual(cand_x) < 1e-3:
                # Tránh trùng lặp
                if not any(np.linalg.norm(cand_x - u) < 1e-3 for u in ga_roots):
                    ga_roots.append(cand_x)

        notes.append(f"GA tìm kiếm đa chiều qua {n_restarts} đợt khởi tạo")

    # Sắp xếp nghiệm
    ga_roots.sort(key=lambda r: float(r[0]))

    elapsed = time.perf_counter() - start_time
    ga_res_obj = GARunResult(
        roots=ga_roots,
        n_evaluations=n_evals,
        elapsed_s=elapsed,
        notes=notes,
        history_best=hist_best,
        history_mean=hist_mean,
    )

    baseline_obj: BaselineResult | None = None
    if req.run_baseline:
        try:
            baseline_obj = solve_baseline(prob, req.bounds)
        except Exception as ex:
            warnings.append(f"Không thể chạy phương pháp cổ điển: {ex}")

    return SolveResponse(
        ok=True,
        problem=prob,
        ga=ga_res_obj,
        baseline=baseline_obj,
        warnings=warnings,
    )


def figure_convergence(resp: SolveResponse, fig_warns: list[str]) -> go.Figure | None:
    """Tạo biểu đồ Plotly thể hiện quá trình hội tụ của GA."""
    if not resp.ga or not resp.ga.history_best:
        fig_warns.append("Không có dữ liệu lịch sử thế hệ để vẽ biểu đồ hội tụ.")
        return None

    best_hist = resp.ga.history_best
    mean_hist = resp.ga.history_mean
    gens = list(range(1, len(best_hist) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens,
        y=best_hist,
        mode="lines",
        name="Best Fitness",
        line=dict(color="#00cc96", width=2.5),
    ))
    if mean_hist and len(mean_hist) == len(best_hist):
        fig.add_trace(go.Scatter(
            x=gens,
            y=mean_hist,
            mode="lines",
            name="Mean Fitness",
            line=dict(color="#ffa15a", width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title="Quá trình hội tụ của Giải thuật Di truyền (GA)",
        xaxis_title="Thế hệ (Generations)",
        yaxis_title="Độ thích nghi (Fitness)",
        template="plotly_dark",
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    return fig


def figure_problem(resp: SolveResponse, B: np.ndarray, fig_warns: list[str]) -> go.Figure | None:
    """Tạo biểu đồ Plotly trực quan hoá hàm số và các nghiệm tìm được."""
    if not resp.problem:
        return None

    prob = resp.problem
    n_vars = len(prob.variables)

    if n_vars == 1:
        v_name = prob.variables[0]
        lo, hi = float(B[0, 0]), float(B[0, 1])
        xs = np.linspace(lo, hi, 600)
        res_arr = prob.residuals(xs.reshape(-1, 1))

        fig = go.Figure()
        # Đường chuẩn y = 0
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)")

        # Vẽ các hàm phương trình
        for i, eq_str in enumerate(prob.equations):
            ys = res_arr[:, i]
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"f{i+1}({v_name})" if len(prob.equations) > 1 else f"f({v_name})",
                line=dict(width=2),
            ))

        # Điểm nghiệm GA
        if resp.ga and resp.ga.roots:
            root_xs = [float(r[0]) for r in resp.ga.roots]
            root_ys = [0.0] * len(root_xs)
            fig.add_trace(go.Scatter(
                x=root_xs,
                y=root_ys,
                mode="markers",
                name="Nghiệm GA",
                marker=dict(color="#ff4b4b", size=11, symbol="circle", line=dict(color="white", width=1.5)),
                hovertext=[f"Nghiệm {v_name} = {rx:.6f}" for rx in root_xs],
            ))

        fig.update_layout(
            title=f"Đồ thị hàm số và vị trí nghiệm ({v_name})",
            xaxis_title=v_name,
            yaxis_title="Giá trị hàm / Phần dư",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=50, b=40),
            height=380,
        )
        return fig

    elif n_vars == 2:
        v1, v2 = prob.variables[0], prob.variables[1]
        x_lo, x_hi = float(B[0, 0]), float(B[0, 1])
        y_lo, y_hi = float(B[1, 0]), float(B[1, 1])

        nx, ny = 80, 80
        xg = np.linspace(x_lo, x_hi, nx)
        yg = np.linspace(y_lo, y_hi, ny)
        Xg, Yg = np.meshgrid(xg, yg)
        grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])

        obj_vals = prob.objective(grid_pts).reshape(Xg.shape)
        log_obj = np.log10(np.clip(obj_vals, 1e-12, 1e6))

        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=xg,
            y=yg,
            z=log_obj,
            colorscale="Viridis",
            contours=dict(coloring="heatmap"),
            colorbar=dict(title="log10(Error)"),
            name="Objective",
        ))

        if resp.ga and resp.ga.roots:
            root_xs = [float(r[0]) for r in resp.ga.roots]
            root_ys = [float(r[1]) for r in resp.ga.roots]
            fig.add_trace(go.Scatter(
                x=root_xs,
                y=root_ys,
                mode="markers",
                name="Nghiệm GA",
                marker=dict(color="#ff2b2b", size=12, symbol="diamond", line=dict(color="white", width=1.5)),
                hovertext=[f"({rx:.4f}, {ry:.4f})" for rx, ry in zip(root_xs, root_ys)],
            ))

        fig.update_layout(
            title=f"Không gian mục tiêu & Vị trí nghiệm ({v1}, {v2})",
            xaxis_title=v1,
            yaxis_title=v2,
            template="plotly_dark",
            margin=dict(l=40, r=40, t=50, b=40),
            height=380,
        )
        return fig

    else:
        fig_warns.append(f"Không hỗ trợ vẽ đồ thị 2D/3D cho hệ có {n_vars} biến.")
        return None
