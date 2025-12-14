from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


TaskKind = Literal[
    "ratio",
    "product",
    "ratio_of_sums",
    "product_of_sums",
    "ratio_diff",
    "product_diff",
    "ratio_x_ratio",
    "product_x_product",
    "ratio_x_product",
    "nonmonotone",
    "gated",
]


@dataclass(frozen=True)
class RatioCoord:
    name: str
    numerator_cols: tuple[int, ...]
    denominator_cols: tuple[int, ...]
    epsilon: float


@dataclass(frozen=True)
class ProductCoord:
    name: str
    group_a_cols: tuple[int, ...]
    group_b_cols: tuple[int, ...]
    epsilon: float


@dataclass(frozen=True)
class TaskDiagnostics:
    ratio_coords: tuple[RatioCoord, ...]
    product_coords: tuple[ProductCoord, ...]
    sum_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TaskTransform:
    coord_names: tuple[str, ...]
    coords: np.ndarray  # (n, n_coords)
    s_names: tuple[str, ...]
    s: np.ndarray  # (n, n_components)
    s_total: np.ndarray  # (n,)


@dataclass(frozen=True)
class FittedTask:
    task_id: str
    level: int
    kind: TaskKind
    component_count: int
    diagnostics: TaskDiagnostics
    gating_threshold: float | None
    nonmonotone_shape: str | None
    nonmonotone_mu: float | None
    nonmonotone_delta: float | None

    def transform(self, x: np.ndarray) -> TaskTransform:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be 2D")

        if self.kind == "ratio":
            return _transform_ratio(self, x)
        if self.kind == "product":
            return _transform_product(self, x)
        if self.kind == "ratio_of_sums":
            return _transform_ratio_of_sums(self, x)
        if self.kind == "product_of_sums":
            return _transform_product_of_sums(self, x)
        if self.kind == "ratio_diff":
            return _transform_ratio_diff(self, x)
        if self.kind == "product_diff":
            return _transform_product_diff(self, x)
        if self.kind == "ratio_x_ratio":
            return _transform_ratio_x_ratio(self, x)
        if self.kind == "product_x_product":
            return _transform_product_x_product(self, x)
        if self.kind == "ratio_x_product":
            return _transform_ratio_x_product(self, x)
        if self.kind == "nonmonotone":
            return _transform_nonmonotone(self, x)
        if self.kind == "gated":
            return _transform_gated(self, x)

        raise ValueError(f"Unknown task kind: {self.kind}")


def _check_signal_budget(required: int, d_signal_max: int) -> None:
    if required > d_signal_max:
        raise ValueError(
            f"Task requires {required} signal columns but d_signal_max={d_signal_max}. "
            "Increase d_signal_max or reduce component_count."
        )


def _median_epsilon(values: np.ndarray, epsilon_rel: float) -> float:
    med = float(np.median(values))
    return float(epsilon_rel) * med


def _log_ratio(num: np.ndarray, denom: np.ndarray, eps: float) -> np.ndarray:
    return np.log(num / (denom + float(eps)))


def _log_product(prod: np.ndarray, eps: float) -> np.ndarray:
    return np.log(prod + float(eps))


def fit_task(
    task_cfg: dict[str, Any],
    x_base: np.ndarray,
    *,
    d_signal_max: int,
    epsilon_rel: float,
    nonmonotone_mu: float,
    nonmonotone_delta: float,
    gating_threshold_quantile: float,
) -> FittedTask:
    task_id = str(task_cfg["id"])
    level = int(task_cfg.get("level", 0))
    kind = task_cfg.get("kind")
    if kind is None:
        raise ValueError(f"Task {task_id} missing 'kind'")
    kind = str(kind)
    component_count = int(task_cfg.get("component_count", 1))
    if component_count <= 0:
        raise ValueError("component_count must be >= 1")

    x_base = np.asarray(x_base, dtype=np.float64)

    if kind == "ratio":
        _check_signal_budget(required=2 * component_count, d_signal_max=d_signal_max)
        ratio_coords = []
        for i in range(component_count):
            num = 2 * i
            denom = 2 * i + 1
            eps = _median_epsilon(x_base[:, denom], epsilon_rel)
            ratio_coords.append(
                RatioCoord(
                    name=f"u_ratio_{i}",
                    numerator_cols=(num,),
                    denominator_cols=(denom,),
                    epsilon=eps,
                )
            )
        return FittedTask(
            task_id=task_id,
            level=level,
            kind="ratio",
            component_count=component_count,
            diagnostics=TaskDiagnostics(
                ratio_coords=tuple(ratio_coords),
                product_coords=tuple(),
                sum_groups=tuple(),
            ),
            gating_threshold=None,
            nonmonotone_shape=None,
            nonmonotone_mu=None,
            nonmonotone_delta=None,
        )

    if kind == "product":
        _check_signal_budget(required=2 * component_count, d_signal_max=d_signal_max)
        product_coords = []
        for i in range(component_count):
            a = 2 * i
            b = 2 * i + 1
            base = x_base[:, a] * x_base[:, b]
            eps = _median_epsilon(base, epsilon_rel)
            product_coords.append(
                ProductCoord(
                    name=f"u_prod_{i}",
                    group_a_cols=(a,),
                    group_b_cols=(b,),
                    epsilon=eps,
                )
            )
        return FittedTask(
            task_id=task_id,
            level=level,
            kind="product",
            component_count=component_count,
            diagnostics=TaskDiagnostics(
                ratio_coords=tuple(),
                product_coords=tuple(product_coords),
                sum_groups=tuple(),
            ),
            gating_threshold=None,
            nonmonotone_shape=None,
            nonmonotone_mu=None,
            nonmonotone_delta=None,
        )

    if kind == "ratio_of_sums":
        _check_signal_budget(required=4 * component_count, d_signal_max=d_signal_max)
        ratio_coords = []
        sum_groups: list[tuple[int, ...]] = []
        for i in range(component_count):
            base_idx = 4 * i
            num_cols = (base_idx + 0, base_idx + 1)
            denom_cols = (base_idx + 2, base_idx + 3)
            denom = x_base[:, denom_cols[0]] + x_base[:, denom_cols[1]]
            eps = _median_epsilon(denom, epsilon_rel)
            ratio_coords.append(
                RatioCoord(
                    name=f"u_ratio_sum_{i}",
                    numerator_cols=num_cols,
                    denominator_cols=denom_cols,
                    epsilon=eps,
                )
            )
            sum_groups.extend([num_cols, denom_cols])
        return FittedTask(
            task_id=task_id,
            level=level,
            kind="ratio_of_sums",
            component_count=component_count,
            diagnostics=TaskDiagnostics(
                ratio_coords=tuple(ratio_coords),
                product_coords=tuple(),
                sum_groups=tuple(sum_groups),
            ),
            gating_threshold=None,
            nonmonotone_shape=None,
            nonmonotone_mu=None,
            nonmonotone_delta=None,
        )

    if kind == "product_of_sums":
        _check_signal_budget(required=4 * component_count, d_signal_max=d_signal_max)
        product_coords = []
        sum_groups: list[tuple[int, ...]] = []
        for i in range(component_count):
            base_idx = 4 * i
            a_cols = (base_idx + 0, base_idx + 1)
            b_cols = (base_idx + 2, base_idx + 3)
            sum_a = x_base[:, a_cols[0]] + x_base[:, a_cols[1]]
            sum_b = x_base[:, b_cols[0]] + x_base[:, b_cols[1]]
            prod = sum_a * sum_b
            eps = _median_epsilon(prod, epsilon_rel)
            product_coords.append(
                ProductCoord(
                    name=f"u_prod_sum_{i}",
                    group_a_cols=a_cols,
                    group_b_cols=b_cols,
                    epsilon=eps,
                )
            )
            sum_groups.extend([a_cols, b_cols])
        return FittedTask(
            task_id=task_id,
            level=level,
            kind="product_of_sums",
            component_count=component_count,
            diagnostics=TaskDiagnostics(
                ratio_coords=tuple(),
                product_coords=tuple(product_coords),
                sum_groups=tuple(sum_groups),
            ),
            gating_threshold=None,
            nonmonotone_shape=None,
            nonmonotone_mu=None,
            nonmonotone_delta=None,
        )

    if kind == "ratio_diff":
        _check_signal_budget(required=4 * component_count, d_signal_max=d_signal_max)
        ratio_coords = []
        for i in range(component_count):
            base_idx = 4 * i
            eps1 = _median_epsilon(x_base[:, base_idx + 1], epsilon_rel)
            eps2 = _median_epsilon(x_base[:, base_idx + 3], epsilon_rel)
            ratio_coords.extend(
                [
                    RatioCoord(
                        name=f"u_ratio_a_{i}",
                        numerator_cols=(base_idx + 0,),
                        denominator_cols=(base_idx + 1,),
                        epsilon=eps1,
                    ),
                    RatioCoord(
                        name=f"u_ratio_b_{i}",
                        numerator_cols=(base_idx + 2,),
                        denominator_cols=(base_idx + 3,),
                        epsilon=eps2,
                    ),
                ]
            )
        return FittedTask(
            task_id=task_id,
            level=level,
            kind="ratio_diff",
            component_count=component_count,
            diagnostics=TaskDiagnostics(
                ratio_coords=tuple(ratio_coords),
                product_coords=tuple(),
                sum_groups=tuple(),
            ),
            gating_threshold=None,
            nonmonotone_shape=None,
            nonmonotone_mu=None,
            nonmonotone_delta=None,
        )

    if kind == "product_diff":
        _check_signal_budget(required=4 * component_count, d_signal_max=d_signal_max)
        product_coords = []
        for i in range(component_count):
            base_idx = 4 * i
            prod1 = x_base[:, base_idx + 0] * x_base[:, base_idx + 1]
            prod2 = x_base[:, base_idx + 2] * x_base[:, base_idx + 3]
            eps1 = _median_epsilon(prod1, epsilon_rel)
            eps2 = _median_epsilon(prod2, epsilon_rel)
            product_coords.extend(
                [
                    ProductCoord(
                        name=f"u_prod_a_{i}",
                        group_a_cols=(base_idx + 0,),
                        group_b_cols=(base_idx + 1,),
                        epsilon=eps1,
                    ),
                    ProductCoord(
                        name=f"u_prod_b_{i}",
                        group_a_cols=(base_idx + 2,),
                        group_b_cols=(base_idx + 3,),
                        epsilon=eps2,
                    ),
                ]
            )
        return FittedTask(
            task_id=task_id,
            level=level,
            kind="product_diff",
            component_count=component_count,
            diagnostics=TaskDiagnostics(
                ratio_coords=tuple(),
                product_coords=tuple(product_coords),
                sum_groups=tuple(),
            ),
            gating_threshold=None,
            nonmonotone_shape=None,
            nonmonotone_mu=None,
            nonmonotone_delta=None,
        )

    if kind == "ratio_x_ratio":
        _check_signal_budget(required=4 * component_count, d_signal_max=d_signal_max)
        ratio_coords = []
        for i in range(component_count):
            base_idx = 4 * i
            eps1 = _median_epsilon(x_base[:, base_idx + 1], epsilon_rel)
            eps2 = _median_epsilon(x_base[:, base_idx + 3], epsilon_rel)
            ratio_coords.extend(
                [
                    RatioCoord(
                        name=f"u1_ratio_{i}",
                        numerator_cols=(base_idx + 0,),
                        denominator_cols=(base_idx + 1,),
                        epsilon=eps1,
                    ),
                    RatioCoord(
                        name=f"u2_ratio_{i}",
                        numerator_cols=(base_idx + 2,),
                        denominator_cols=(base_idx + 3,),
                        epsilon=eps2,
                    ),
                ]
            )
        return FittedTask(
            task_id=task_id,
            level=level,
            kind="ratio_x_ratio",
            component_count=component_count,
            diagnostics=TaskDiagnostics(
                ratio_coords=tuple(ratio_coords),
                product_coords=tuple(),
                sum_groups=tuple(),
            ),
            gating_threshold=None,
            nonmonotone_shape=None,
            nonmonotone_mu=None,
            nonmonotone_delta=None,
        )

    if kind == "product_x_product":
        _check_signal_budget(required=4 * component_count, d_signal_max=d_signal_max)
        product_coords = []
        for i in range(component_count):
            base_idx = 4 * i
            prod1 = x_base[:, base_idx + 0] * x_base[:, base_idx + 1]
            prod2 = x_base[:, base_idx + 2] * x_base[:, base_idx + 3]
            eps1 = _median_epsilon(prod1, epsilon_rel)
            eps2 = _median_epsilon(prod2, epsilon_rel)
            product_coords.extend(
                [
                    ProductCoord(
                        name=f"u1_prod_{i}",
                        group_a_cols=(base_idx + 0,),
                        group_b_cols=(base_idx + 1,),
                        epsilon=eps1,
                    ),
                    ProductCoord(
                        name=f"u2_prod_{i}",
                        group_a_cols=(base_idx + 2,),
                        group_b_cols=(base_idx + 3,),
                        epsilon=eps2,
                    ),
                ]
            )
        return FittedTask(
            task_id=task_id,
            level=level,
            kind="product_x_product",
            component_count=component_count,
            diagnostics=TaskDiagnostics(
                ratio_coords=tuple(),
                product_coords=tuple(product_coords),
                sum_groups=tuple(),
            ),
            gating_threshold=None,
            nonmonotone_shape=None,
            nonmonotone_mu=None,
            nonmonotone_delta=None,
        )

    if kind == "ratio_x_product":
        _check_signal_budget(required=4 * component_count, d_signal_max=d_signal_max)
        ratio_coords = []
        product_coords = []
        for i in range(component_count):
            base_idx = 4 * i
            eps_ratio = _median_epsilon(x_base[:, base_idx + 1], epsilon_rel)
            prod = x_base[:, base_idx + 2] * x_base[:, base_idx + 3]
            eps_prod = _median_epsilon(prod, epsilon_rel)
            ratio_coords.append(
                RatioCoord(
                    name=f"u_ratio_{i}",
                    numerator_cols=(base_idx + 0,),
                    denominator_cols=(base_idx + 1,),
                    epsilon=eps_ratio,
                )
            )
            product_coords.append(
                ProductCoord(
                    name=f"u_prod_{i}",
                    group_a_cols=(base_idx + 2,),
                    group_b_cols=(base_idx + 3,),
                    epsilon=eps_prod,
                )
            )
        return FittedTask(
            task_id=task_id,
            level=level,
            kind="ratio_x_product",
            component_count=component_count,
            diagnostics=TaskDiagnostics(
                ratio_coords=tuple(ratio_coords),
                product_coords=tuple(product_coords),
                sum_groups=tuple(),
            ),
            gating_threshold=None,
            nonmonotone_shape=None,
            nonmonotone_mu=None,
            nonmonotone_delta=None,
        )

    if kind == "nonmonotone":
        base = str(task_cfg.get("base", "l1_ratio"))
        shape = str(task_cfg.get("shape", "u_shaped"))
        base_kind: TaskKind
        if base in {"l1_ratio", "ratio"}:
            base_kind = "ratio"
        elif base in {"l1_product", "product"}:
            base_kind = "product"
        else:
            raise ValueError(f"Unsupported nonmonotone base: {base}")

        # Fit a single base coordinate spec; transform will re-use it.
        fitted_base = fit_task(
            {"id": f"{task_id}__base", "level": level, "kind": base_kind, "component_count": component_count},
            x_base,
            d_signal_max=d_signal_max,
            epsilon_rel=epsilon_rel,
            nonmonotone_mu=nonmonotone_mu,
            nonmonotone_delta=nonmonotone_delta,
            gating_threshold_quantile=gating_threshold_quantile,
        )

        return FittedTask(
            task_id=task_id,
            level=level,
            kind="nonmonotone",
            component_count=component_count,
            diagnostics=fitted_base.diagnostics,
            gating_threshold=None,
            nonmonotone_shape=shape,
            nonmonotone_mu=float(nonmonotone_mu),
            nonmonotone_delta=float(nonmonotone_delta),
        )

    if kind == "gated":
        if component_count != 1:
            raise ValueError("gated tasks currently require component_count=1")
        ratio_task = str(task_cfg.get("ratio_task", "l1_ratio"))
        product_task = str(task_cfg.get("product_task", "l1_product"))
        if ratio_task not in {"l1_ratio", "ratio"}:
            raise ValueError(f"Unsupported gated ratio_task: {ratio_task}")
        if product_task not in {"l1_product", "product"}:
            raise ValueError(f"Unsupported gated product_task: {product_task}")
        _check_signal_budget(required=5, d_signal_max=d_signal_max)

        fitted_ratio = fit_task(
            {"id": f"{task_id}__ratio", "level": level, "kind": "ratio", "component_count": 1},
            x_base,
            d_signal_max=d_signal_max,
            epsilon_rel=epsilon_rel,
            nonmonotone_mu=nonmonotone_mu,
            nonmonotone_delta=nonmonotone_delta,
            gating_threshold_quantile=gating_threshold_quantile,
        )
        fitted_prod = fit_task(
            {"id": f"{task_id}__prod", "level": level, "kind": "product", "component_count": 1},
            x_base,
            d_signal_max=d_signal_max,
            epsilon_rel=epsilon_rel,
            nonmonotone_mu=nonmonotone_mu,
            nonmonotone_delta=nonmonotone_delta,
            gating_threshold_quantile=gating_threshold_quantile,
        )

        g = x_base[:, 4]
        q = float(gating_threshold_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("gating.threshold_quantile must be in (0,1)")
        t = float(np.quantile(g, q))

        return FittedTask(
            task_id=task_id,
            level=level,
            kind="gated",
            component_count=1,
            diagnostics=TaskDiagnostics(
                ratio_coords=fitted_ratio.diagnostics.ratio_coords,
                product_coords=fitted_prod.diagnostics.product_coords,
                sum_groups=tuple(),
            ),
            gating_threshold=t,
            nonmonotone_shape=None,
            nonmonotone_mu=None,
            nonmonotone_delta=None,
        )

    raise ValueError(f"Unknown or unsupported task kind: {kind}")


def _transform_ratio(task: FittedTask, x: np.ndarray) -> TaskTransform:
    ratio_coords = task.diagnostics.ratio_coords
    coords = []
    s_cols = []
    for rc in ratio_coords:
        num = x[:, rc.numerator_cols].sum(axis=1)
        denom = x[:, rc.denominator_cols].sum(axis=1)
        u = _log_ratio(num, denom, rc.epsilon)
        coords.append(u)
        s_cols.append(u)
    coords_m = np.stack(coords, axis=1) if coords else np.zeros((x.shape[0], 0), dtype=np.float64)
    s_m = np.stack(s_cols, axis=1)
    return TaskTransform(
        coord_names=tuple(rc.name for rc in ratio_coords),
        coords=coords_m,
        s_names=tuple(f"s_{i}" for i in range(task.component_count)),
        s=s_m,
        s_total=s_m.sum(axis=1),
    )


def _transform_product(task: FittedTask, x: np.ndarray) -> TaskTransform:
    product_coords = task.diagnostics.product_coords
    coords = []
    s_cols = []
    for pc in product_coords:
        a = x[:, pc.group_a_cols].prod(axis=1)
        b = x[:, pc.group_b_cols].prod(axis=1)
        u = _log_product(a * b, pc.epsilon)
        coords.append(u)
        s_cols.append(u)
    coords_m = np.stack(coords, axis=1) if coords else np.zeros((x.shape[0], 0), dtype=np.float64)
    s_m = np.stack(s_cols, axis=1)
    return TaskTransform(
        coord_names=tuple(pc.name for pc in product_coords),
        coords=coords_m,
        s_names=tuple(f"s_{i}" for i in range(task.component_count)),
        s=s_m,
        s_total=s_m.sum(axis=1),
    )


def _transform_ratio_of_sums(task: FittedTask, x: np.ndarray) -> TaskTransform:
    ratio_coords = task.diagnostics.ratio_coords
    coords = []
    s_cols = []
    for rc in ratio_coords:
        num = x[:, rc.numerator_cols].sum(axis=1)
        denom = x[:, rc.denominator_cols].sum(axis=1)
        u = _log_ratio(num, denom, rc.epsilon)
        coords.append(u)
        s_cols.append(u)
    coords_m = np.stack(coords, axis=1)
    s_m = np.stack(s_cols, axis=1)
    return TaskTransform(
        coord_names=tuple(rc.name for rc in ratio_coords),
        coords=coords_m,
        s_names=tuple(f"s_{i}" for i in range(task.component_count)),
        s=s_m,
        s_total=s_m.sum(axis=1),
    )


def _transform_product_of_sums(task: FittedTask, x: np.ndarray) -> TaskTransform:
    product_coords = task.diagnostics.product_coords
    coords = []
    s_cols = []
    for pc in product_coords:
        sum_a = x[:, pc.group_a_cols].sum(axis=1)
        sum_b = x[:, pc.group_b_cols].sum(axis=1)
        u = _log_product(sum_a * sum_b, pc.epsilon)
        coords.append(u)
        s_cols.append(u)
    coords_m = np.stack(coords, axis=1)
    s_m = np.stack(s_cols, axis=1)
    return TaskTransform(
        coord_names=tuple(pc.name for pc in product_coords),
        coords=coords_m,
        s_names=tuple(f"s_{i}" for i in range(task.component_count)),
        s=s_m,
        s_total=s_m.sum(axis=1),
    )


def _transform_ratio_diff(task: FittedTask, x: np.ndarray) -> TaskTransform:
    ratio_coords = task.diagnostics.ratio_coords
    if len(ratio_coords) != 2 * task.component_count:
        raise AssertionError("ratio_diff expects 2 ratio coords per component")

    coords_cols = []
    s_cols = []
    for i in range(task.component_count):
        a = ratio_coords[2 * i]
        b = ratio_coords[2 * i + 1]
        ua = _log_ratio(
            x[:, a.numerator_cols].sum(axis=1), x[:, a.denominator_cols].sum(axis=1), a.epsilon
        )
        ub = _log_ratio(
            x[:, b.numerator_cols].sum(axis=1), x[:, b.denominator_cols].sum(axis=1), b.epsilon
        )
        coords_cols.extend([ua, ub])
        s_cols.append(ua - ub)
    coords_m = np.stack(coords_cols, axis=1)
    s_m = np.stack(s_cols, axis=1)
    return TaskTransform(
        coord_names=tuple(rc.name for rc in ratio_coords),
        coords=coords_m,
        s_names=tuple(f"s_{i}" for i in range(task.component_count)),
        s=s_m,
        s_total=s_m.sum(axis=1),
    )


def _transform_product_diff(task: FittedTask, x: np.ndarray) -> TaskTransform:
    product_coords = task.diagnostics.product_coords
    if len(product_coords) != 2 * task.component_count:
        raise AssertionError("product_diff expects 2 product coords per component")

    coords_cols = []
    s_cols = []
    for i in range(task.component_count):
        a = product_coords[2 * i]
        b = product_coords[2 * i + 1]
        ua = _log_product(
            x[:, a.group_a_cols].prod(axis=1) * x[:, a.group_b_cols].prod(axis=1), a.epsilon
        )
        ub = _log_product(
            x[:, b.group_a_cols].prod(axis=1) * x[:, b.group_b_cols].prod(axis=1), b.epsilon
        )
        coords_cols.extend([ua, ub])
        s_cols.append(ua - ub)
    coords_m = np.stack(coords_cols, axis=1)
    s_m = np.stack(s_cols, axis=1)
    return TaskTransform(
        coord_names=tuple(pc.name for pc in product_coords),
        coords=coords_m,
        s_names=tuple(f"s_{i}" for i in range(task.component_count)),
        s=s_m,
        s_total=s_m.sum(axis=1),
    )


def _transform_ratio_x_ratio(task: FittedTask, x: np.ndarray) -> TaskTransform:
    ratio_coords = task.diagnostics.ratio_coords
    if len(ratio_coords) != 2 * task.component_count:
        raise AssertionError("ratio_x_ratio expects 2 ratio coords per component")

    coords_cols = []
    s_cols = []
    for i in range(task.component_count):
        u1 = ratio_coords[2 * i]
        u2 = ratio_coords[2 * i + 1]
        a = _log_ratio(
            x[:, u1.numerator_cols].sum(axis=1), x[:, u1.denominator_cols].sum(axis=1), u1.epsilon
        )
        b = _log_ratio(
            x[:, u2.numerator_cols].sum(axis=1), x[:, u2.denominator_cols].sum(axis=1), u2.epsilon
        )
        coords_cols.extend([a, b])
        s_cols.append(a * b)
    coords_m = np.stack(coords_cols, axis=1)
    s_m = np.stack(s_cols, axis=1)
    return TaskTransform(
        coord_names=tuple(rc.name for rc in ratio_coords),
        coords=coords_m,
        s_names=tuple(f"s_{i}" for i in range(task.component_count)),
        s=s_m,
        s_total=s_m.sum(axis=1),
    )


def _transform_product_x_product(task: FittedTask, x: np.ndarray) -> TaskTransform:
    product_coords = task.diagnostics.product_coords
    if len(product_coords) != 2 * task.component_count:
        raise AssertionError("product_x_product expects 2 product coords per component")

    coords_cols = []
    s_cols = []
    for i in range(task.component_count):
        u1 = product_coords[2 * i]
        u2 = product_coords[2 * i + 1]
        a = _log_product(
            x[:, u1.group_a_cols].prod(axis=1) * x[:, u1.group_b_cols].prod(axis=1), u1.epsilon
        )
        b = _log_product(
            x[:, u2.group_a_cols].prod(axis=1) * x[:, u2.group_b_cols].prod(axis=1), u2.epsilon
        )
        coords_cols.extend([a, b])
        s_cols.append(a * b)
    coords_m = np.stack(coords_cols, axis=1)
    s_m = np.stack(s_cols, axis=1)
    return TaskTransform(
        coord_names=tuple(pc.name for pc in product_coords),
        coords=coords_m,
        s_names=tuple(f"s_{i}" for i in range(task.component_count)),
        s=s_m,
        s_total=s_m.sum(axis=1),
    )


def _transform_ratio_x_product(task: FittedTask, x: np.ndarray) -> TaskTransform:
    ratio_coords = task.diagnostics.ratio_coords
    product_coords = task.diagnostics.product_coords
    if len(ratio_coords) != task.component_count:
        raise AssertionError("ratio_x_product expects 1 ratio coord per component")
    if len(product_coords) != task.component_count:
        raise AssertionError("ratio_x_product expects 1 product coord per component")

    coords_cols = []
    s_cols = []
    for i in range(task.component_count):
        r = ratio_coords[i]
        p = product_coords[i]
        u1 = _log_ratio(x[:, r.numerator_cols].sum(axis=1), x[:, r.denominator_cols].sum(axis=1), r.epsilon)
        u2 = _log_product(
            x[:, p.group_a_cols].prod(axis=1) * x[:, p.group_b_cols].prod(axis=1), p.epsilon
        )
        coords_cols.extend([u1, u2])
        s_cols.append(u1 * u2)
    coords_m = np.stack(coords_cols, axis=1)
    s_m = np.stack(s_cols, axis=1)
    coord_names = []
    for i in range(task.component_count):
        coord_names.extend([ratio_coords[i].name, product_coords[i].name])
    return TaskTransform(
        coord_names=tuple(coord_names),
        coords=coords_m,
        s_names=tuple(f"s_{i}" for i in range(task.component_count)),
        s=s_m,
        s_total=s_m.sum(axis=1),
    )


def _transform_nonmonotone(task: FittedTask, x: np.ndarray) -> TaskTransform:
    mu = float(task.nonmonotone_mu or 0.0)
    delta = float(task.nonmonotone_delta or 1.0)
    if delta <= 0:
        raise ValueError("nonmonotone delta must be > 0")
    shape = str(task.nonmonotone_shape or "u_shaped")

    # Base coordinate(s) are stored in diagnostics; compute as normal, then apply shape to the summed u.
    if task.diagnostics.ratio_coords:
        base = _transform_ratio(
            FittedTask(
                task_id=task.task_id,
                level=task.level,
                kind="ratio",
                component_count=task.component_count,
                diagnostics=task.diagnostics,
                gating_threshold=None,
                nonmonotone_shape=None,
                nonmonotone_mu=None,
                nonmonotone_delta=None,
            ),
            x,
        )
        u_total = base.s_total
        coords_m = base.coords
        coord_names = base.coord_names
    elif task.diagnostics.product_coords:
        base = _transform_product(
            FittedTask(
                task_id=task.task_id,
                level=task.level,
                kind="product",
                component_count=task.component_count,
                diagnostics=task.diagnostics,
                gating_threshold=None,
                nonmonotone_shape=None,
                nonmonotone_mu=None,
                nonmonotone_delta=None,
            ),
            x,
        )
        u_total = base.s_total
        coords_m = base.coords
        coord_names = base.coord_names
    else:
        raise AssertionError("nonmonotone must be based on ratio or product coords")

    if shape in {"u_shaped", "peak"}:
        s_total = -((u_total - mu) / delta) ** 2
    elif shape in {"band_pass", "gaussian"}:
        s_total = np.exp(-((u_total - mu) ** 2) / (2.0 * (delta**2)))
    else:
        raise ValueError(f"Unknown nonmonotone shape: {shape}")

    s_m = s_total.reshape(-1, 1)
    return TaskTransform(
        coord_names=coord_names,
        coords=coords_m,
        s_names=("s_0",),
        s=s_m,
        s_total=s_total,
    )


def _transform_gated(task: FittedTask, x: np.ndarray) -> TaskTransform:
    if task.gating_threshold is None:
        raise ValueError("gated task missing gating_threshold")
    t = float(task.gating_threshold)
    g = x[:, 4]
    I = (g > t).astype(np.float64, copy=False)

    # We assume diagnostics carries exactly one ratio coord and one product coord.
    if len(task.diagnostics.ratio_coords) != 1:
        raise AssertionError("gated expects exactly one ratio coord")
    if len(task.diagnostics.product_coords) != 1:
        raise AssertionError("gated expects exactly one product coord")

    rc = task.diagnostics.ratio_coords[0]
    pc = task.diagnostics.product_coords[0]
    u_ratio = _log_ratio(
        x[:, rc.numerator_cols].sum(axis=1),
        x[:, rc.denominator_cols].sum(axis=1),
        rc.epsilon,
    )
    u_prod = _log_product(
        x[:, pc.group_a_cols].prod(axis=1) * x[:, pc.group_b_cols].prod(axis=1),
        pc.epsilon,
    )

    s_total = I * u_ratio + (1.0 - I) * u_prod
    coords_m = np.stack([I, u_ratio, u_prod], axis=1)
    return TaskTransform(
        coord_names=("gate_I", rc.name, pc.name),
        coords=coords_m,
        s_names=("s_0",),
        s=s_total.reshape(-1, 1),
        s_total=s_total,
    )
