from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np
import pandas as pd

__all__ = ["evaluate_drawdown_control"]


def _day(value, name):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"{name}: нужна дата.")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    if stamp != stamp.normalize():
        raise ValueError(f"{name} должен обозначать начало дня (00:00 UTC).")
    return stamp


def _parameters(q, k, recovery_fraction, reduced_multiplier):
    params = dict(q=q, k=k, recovery_fraction=recovery_fraction,
                  reduced_multiplier=reduced_multiplier)
    for key, value in params.items():
        if not np.isscalar(value) or isinstance(value, (bool, str)):
            raise ValueError(f"{key}: требуется конечное число.")
        try:
            params[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key}: требуется конечное число.") from None
        if not np.isfinite(params[key]):
            raise ValueError(f"{key}: требуется конечное число.")
    if not 0 < params["q"] < 1:
        raise ValueError("q должен находиться между 0 и 1, например 0.95.")
    if params["k"] <= 0:
        raise ValueError("k должен быть положительным.")
    if not 0 < params["recovery_fraction"] <= 1:
        raise ValueError("recovery_fraction должен находиться в (0, 1].")
    if not 0 <= params["reduced_multiplier"] <= 1:
        raise ValueError("reduced_multiplier должен находиться в [0, 1].")
    return params


def _thresholds(history, method, params, policy, label):
    if method == "maxDD":
        threshold = params["k"] * history.max()
    else:
        dar = history.quantile(params["q"])
        threshold = (dar if method == "DaR" else
                     history.where(history.ge(dar, axis="columns")).mean())
    valid = np.isfinite(threshold) & threshold.gt(0)
    if not valid.all():
        names = ", ".join(map(str, threshold.index[~valid][:8]))
        message = (f"{label}: нет положительного порога по истории до evaluation_start "
                   f"для {names}. Нужна история с просадками; данные теста не используются.")
        if policy == "raise":
            raise ValueError(message + " Можно явно задать on_insufficient_history='unrestricted'.")
        warnings.warn(message + " Эти объекты остаются без ограничения.", UserWarning,
                      stacklevel=3)
    # inf означает явное отсутствие сигнала, а не порог, полученный из теста.
    return threshold.where(valid, np.inf), valid


def _restriction_flags(depth, thresholds, start, params):
    values = depth.shift(1, fill_value=0.0).to_numpy()
    threshold = thresholds.to_numpy()
    enabled = np.isfinite(threshold)
    restore = np.zeros_like(threshold)
    restore[enabled] = (1 - params["recovery_fraction"]) * threshold[enabled]
    flags = np.zeros(values.shape, dtype=bool)
    state = np.zeros(values.shape[1], dtype=bool)
    for pos in np.flatnonzero(depth.index >= start):
        current = values[pos]
        state = np.where(
            enabled & (current >= threshold), True,
            np.where(enabled & (current <= restore + 1e-10), False, state),
        )
        flags[pos] = state
    return pd.DataFrame(flags, index=depth.index, columns=depth.columns)


def _tail_ci(depth, resamples, block_days, confidence, seed):
    values = depth.to_numpy(dtype=float)
    size = len(values)
    if size < 2:
        return (np.nan,) * 4
    block = min(block_days, max(1, size // 4))
    count = int(np.ceil(size / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, size, size=(resamples, count))
    statistics = np.empty((resamples, 2))
    for offset in range(0, resamples, 128):
        selected = starts[offset:offset + 128]
        indices = ((selected[:, :, None] + np.arange(block)) % size).reshape(len(selected), -1)[:, :size]
        sample = values[indices]
        dar = np.quantile(sample, 0.95, axis=1)
        tail = sample >= dar[:, None]
        cdar = np.where(tail, sample, 0).sum(axis=1) / tail.sum(axis=1)
        statistics[offset:offset + len(selected)] = np.column_stack([dar, cdar])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(statistics, [alpha, 1 - alpha], axis=0)
    return low[0], high[0], low[1], high[1]


def _equity_path(events, initial, start, end):
    opening = float(initial + events.loc[events.index < start].sum())
    selected = events.loc[(events.index >= start) & (events.index < end + pd.Timedelta(days=1))]
    # Снимок ДО первого события: даже сделка ровно в 00:00 не потеряется.
    return pd.concat([
        pd.Series([opening], index=[start - pd.Timedelta(nanoseconds=1)]),
        opening + selected.cumsum(),
    ])


def _metrics(events, daily, initial, data, multiplier, switches, start, end,
             serious_threshold, ci_options=None):
    equity = _equity_path(events, initial, start, end)
    if equity.le(0).any():
        raise ValueError("Денежный баланс достиг нуля или стал отрицательным: стандартные метрики доходности неприменимы.")
    opening, final = equity.iloc[0], equity.iloc[-1]
    balance = (initial + daily.cumsum()).loc[start:end]
    previous = (initial + daily.cumsum()).shift(1, fill_value=initial)
    returns = (daily / previous).loc[start:end]
    peak = balance.cummax().clip(lower=opening)
    depth = ((1 - balance / peak) * 100).clip(lower=0)
    cash_depth = (peak - balance).clip(lower=0)
    event_peak = equity.cummax()
    mdd = float(((1 - equity / event_peak) * 100).max())
    cash_mdd = float((event_peak - equity).max())
    underwater = cash_depth > 1e-9
    blocks = (~underwater).cumsum()
    days = len(balance)
    net = final - opening
    cagr = ((final / opening) ** (365.25 / days) - 1) * 100
    std = returns.std() if days > 1 else np.nan
    downside = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
    dar, cash_dar = depth.quantile(0.95), cash_depth.quantile(0.95)
    entries = multiplier[data["_entry_day"].between(start, end).to_numpy()]
    row = {
        "Start_USDT": opening, "End_USDT": final, "Net_PnL_USDT": net,
        "Return_pct": (final / opening - 1) * 100, "CAGR_pct": cagr,
        "MDD_pct": mdd, "Max_drawdown_USDT": cash_mdd,
        "DaR95_pct": dar, "CDaR95_pct": depth[depth >= dar].mean(),
        "DaR95_USDT": cash_dar, "CDaR95_USDT": cash_depth[cash_depth >= cash_dar].mean(),
        "Calmar": cagr / mdd if mdd > 0 else np.nan,
        "RecoveryFactor": net / cash_mdd if cash_mdd > 0 else np.nan,
        "Sharpe": returns.mean() / std * np.sqrt(365) if std > 0 else np.nan,
        "Sortino": returns.mean() / downside * np.sqrt(365) if downside > 0 else np.nan,
        "Max_DD_duration_days": int(underwater.groupby(blocks).sum().max()),
        "Time_underwater_pct": underwater.mean() * 100,
        "Serious_DD_count": int(depth.groupby(blocks).max().gt(serious_threshold + 1e-10).sum()),
        "Serious_DD_days": int(depth.gt(serious_threshold + 1e-10).sum()),
        "Avg_multiplier": entries.mean() if entries.size else np.nan,
        "Trades_half_pct": np.mean(entries == 0.5) * 100 if entries.size else np.nan,
        "Trades_skipped_pct": np.mean(entries == 0) * 100 if entries.size else np.nan,
        "Trades_reduced_pct": np.mean((entries > 0) & (entries < 1)) * 100 if entries.size else np.nan,
        "Switches": int(switches.loc[start:end].sum()),
    }
    if ci_options is not None:
        for unit, series in [("pct", depth), ("USDT", cash_depth)]:
            lo_d, hi_d, lo_c, hi_c = _tail_ci(series, **ci_options)
            row.update({f"DaR95_{unit}_CI_low": lo_d, f"DaR95_{unit}_CI_high": hi_d,
                        f"CDaR95_{unit}_CI_low": lo_c, f"CDaR95_{unit}_CI_high": hi_c})
    return row


def _comparison(rows):
    frame = pd.DataFrame.from_dict(rows, orient="index").rename_axis("Method")
    baseline = frame.loc["Baseline", "Net_PnL_USDT"]
    frame["PnL_delta_USDT"] = frame["Net_PnL_USDT"] - baseline
    frame["PnL_retained_pct"] = frame["Net_PnL_USDT"] / baseline * 100 if baseline > 0 else np.nan
    return frame


def evaluate_drawdown_control(
    trades: pd.DataFrame,
    method: str,
    scope: str,
    evaluation_start,
    *,
    q: float = 0.95,
    k: float = 1.0,
    recovery_fraction: float = 0.5,
    reduced_multiplier: float = 0.0,
    initial_capital: float | None = None,
    evaluation_end=None,
    portfolio_hyperparameters: Mapping | None = None,
    on_insufficient_history: str = "raise",
    serious_dd_threshold_pct: float = 3.0,
    compute_ci: bool = True,
    ci_resamples: int = 2000,
    ci_block_days: int = 30,
    ci_confidence: float = 0.95,
    random_state: int = 42,
    plot: bool = True,
    show: bool = True,
) -> dict:
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        raise ValueError("trades должен быть непустым DataFrame.")
    methods = {"maxdd": "maxDD", "mdd": "maxDD", "dar": "DaR", "cdar": "CDaR"}
    method = methods.get(str(method).lower())
    if method is None:
        raise ValueError("method: maxDD (или MDD), DaR либо CDaR.")
    if scope not in {"individual", "portfolio", "both"}:
        raise ValueError("scope: individual, portfolio либо both.")
    if on_insufficient_history not in {"raise", "unrestricted"}:
        raise ValueError("on_insufficient_history: raise либо unrestricted.")
    params = _parameters(q, k, recovery_fraction, reduced_multiplier)
    portfolio_params = dict(params)
    if portfolio_hyperparameters is not None:
        if scope != "both" or not isinstance(portfolio_hyperparameters, Mapping):
            raise ValueError("portfolio_hyperparameters — словарь только для scope='both'.")
        if set(portfolio_hyperparameters) - set(params):
            raise ValueError("Неизвестные ключи portfolio_hyperparameters.")
        portfolio_params.update(portfolio_hyperparameters)
        portfolio_params = _parameters(**portfolio_params)
    if not np.isfinite(serious_dd_threshold_pct) or serious_dd_threshold_pct <= 0:
        raise ValueError("serious_dd_threshold_pct должен быть положительным.")
    if (not isinstance(ci_resamples, (int, np.integer)) or ci_resamples < 2 or
            not isinstance(ci_block_days, (int, np.integer)) or ci_block_days < 1 or
            not 0 < ci_confidence < 1 or not isinstance(random_state, (int, np.integer)) or random_state < 0):
        raise ValueError("Неверные параметры бутстрэпа (B>=2, block>=1, 0<confidence<1, seed>=0).")

    required = {"entry_date", "exit_date", "profit_abs"}
    individual = scope in {"individual", "both"}
    if individual:
        required |= {"strategy_name", "profit_eq_pc"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"В trades отсутствуют столбцы: {sorted(missing)}.")
    if not trades.columns.is_unique:
        raise ValueError("Названия столбцов trades должны быть уникальны.")
    data = trades.copy(deep=True).reset_index(drop=True)
    for column in ["entry_date", "exit_date"]:
        data[column] = pd.to_datetime(data[column], errors="raise", utc=True).dt.tz_localize(None)
    if data[["entry_date", "exit_date"]].isna().any().any():
        raise ValueError("Передайте закрытые сделки без пустых дат entry_date/exit_date.")
    if (data["exit_date"] < data["entry_date"]).any():
        raise ValueError("exit_date не может предшествовать entry_date.")
    for column in ["profit_abs"] + (["profit_eq_pc"] if individual else []):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} содержит NaN или бесконечность.")
    if "strategy_name" not in data:
        data["strategy_name"] = "Portfolio"
    if data["strategy_name"].isna().any() or not data["strategy_name"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("strategy_name должен содержать непустые строковые идентификаторы.")
    data["_entry_day"] = data["entry_date"].dt.normalize()
    data["_exit_day"] = data["exit_date"].dt.normalize()
    start = _day(evaluation_start, "evaluation_start")
    end = _day(evaluation_end, "evaluation_end") if evaluation_end is not None else data["_exit_day"].max()
    first = data["_entry_day"].min()
    if not first < start <= end:
        raise ValueError("Нужна история до evaluation_start и evaluation_end >= evaluation_start.")
    if initial_capital is None:
        if "start_equity" not in data:
            raise ValueError("Укажите initial_capital — общий капитал на начало истории.")
        first_group = data.loc[data.entry_date.eq(data.entry_date.min()), "start_equity"]
        capital = pd.to_numeric(first_group, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(capital).all() or not np.allclose(capital, capital[0], rtol=1e-10, atol=1e-8):
            raise ValueError("Неоднозначный начальный баланс: укажите initial_capital явно.")
        initial_capital = float(capital[0])
    if not np.isscalar(initial_capital) or isinstance(initial_capital, (bool, str)):
        raise ValueError("initial_capital должен быть положительным конечным числом.")
    try:
        initial_capital = float(initial_capital)
    except (TypeError, ValueError):
        raise ValueError("initial_capital должен быть положительным конечным числом.") from None
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital должен быть положительным конечным числом.")
    calendar = pd.date_range(first, end, freq="D", name="date")
    strategies = pd.Index(sorted(data.loc[data["_entry_day"] <= end, "strategy_name"].unique()), name="strategy_name")
    baseline_events = data.groupby("exit_date")["profit_abs"].sum().sort_index()
    baseline_daily = data.groupby("_exit_day")["profit_abs"].sum().reindex(calendar, fill_value=0.0)
    baseline_balance = initial_capital + baseline_daily.cumsum()
    baseline_path = initial_capital + baseline_events.loc[baseline_events.index < end + pd.Timedelta(days=1)].cumsum()
    if baseline_path.le(0).any():
        raise ValueError("Baseline имеет неположительный реализованный баланс. Проверьте капитал и данные.")
    multiplier = np.ones(len(data))
    entry_days = calendar.get_indexer(data["_entry_day"])
    entry_strategies = strategies.get_indexer(data["strategy_name"])
    in_calendar = (entry_days >= 0) & (entry_strategies >= 0)
    modes = {}
    threshold_tables = []

    if individual:
        virtual = data.groupby(["_exit_day", "strategy_name"])["profit_eq_pc"].sum().unstack("strategy_name")
        virtual = virtual.reindex(index=calendar, columns=strategies).fillna(0).cumsum()
        depth = virtual.cummax().clip(lower=0) - virtual
        first_entry = data.groupby("strategy_name")["_entry_day"].min().reindex(strategies)
        history = depth.loc[depth.index < start]
        active = history.index.to_numpy()[:, None] >= first_entry.to_numpy()[None, :]
        history = history.where(active)
        threshold, valid = _thresholds(history, method, params, on_insufficient_history, "Individual")
        flags = _restriction_flags(depth, threshold, start, params)
        mode = pd.DataFrame(np.where(flags, params["reduced_multiplier"], 1.0), index=calendar, columns=strategies)
        modes["individual"] = mode
        multiplier[in_calendar] = mode.to_numpy()[entry_days[in_calendar], entry_strategies[in_calendar]]
        table = pd.DataFrame({"Threshold": threshold.where(valid), "Unit": "pp (profit_eq_pc)",
                              "Restore_depth": ((1 - params["recovery_fraction"]) * threshold).where(valid),
                              "Calibration_days": history.count(), "Control_enabled": valid})
        threshold_tables.append(pd.concat({"individual": table}, names=["Level", "Object"]))

    if scope in {"portfolio", "both"}:
        depth = ((1 - baseline_balance / baseline_balance.cummax().clip(lower=initial_capital)) * 100).to_frame("Portfolio")
        history = depth.loc[depth.index < start]
        threshold, valid = _thresholds(history, method, portfolio_params, on_insufficient_history, "Portfolio")
        flags = _restriction_flags(depth, threshold, start, portfolio_params)
        mode = pd.Series(np.where(flags.iloc[:, 0], portfolio_params["reduced_multiplier"], 1.0), index=calendar, name="Portfolio")
        modes["portfolio"] = mode
        portfolio_at_entry = np.ones(len(data))
        portfolio_at_entry[in_calendar] = mode.to_numpy()[entry_days[in_calendar]]
        multiplier = np.minimum(multiplier, portfolio_at_entry)
        table = pd.DataFrame({"Threshold": threshold.where(valid), "Unit": "pct (cash equity)",
                              "Restore_depth": ((1 - portfolio_params["recovery_fraction"]) * threshold).where(valid),
                              "Calibration_days": history.count(), "Control_enabled": valid})
        threshold_tables.append(pd.concat({"portfolio": table}, names=["Level", "Object"]))

    if scope == "portfolio":
        effective = modes["portfolio"].to_frame()
    else:
        effective = modes["individual"].copy()
        if scope == "both":
            effective[:] = np.minimum(effective.to_numpy(), modes["portfolio"].to_numpy()[:, None])
    modes["effective"] = effective
    switches = effective.ne(effective.shift(1, fill_value=1.0)).sum(axis=1)
    baseline_switches = pd.Series(0, index=calendar)
    controlled_values = data["profit_abs"].to_numpy() * multiplier
    controlled_events = pd.Series(controlled_values, index=data.index).groupby(data["exit_date"]).sum().sort_index()
    controlled_daily = pd.Series(controlled_values, index=data.index).groupby(data["_exit_day"]).sum().reindex(calendar, fill_value=0)
    label = f"{scope} {method}"
    runs = {"Baseline": (baseline_events, baseline_daily, np.ones(len(data)), baseline_switches),
            label: (controlled_events, controlled_daily, multiplier, switches)}
    ci_options = dict(resamples=ci_resamples, block_days=ci_block_days,
                      confidence=ci_confidence, seed=random_state) if compute_ci else None
    def evaluate_period(begin, finish, with_ci=False):
        return _comparison({name: _metrics(events, daily, initial_capital, data, weights, changes,
                                          begin, finish, serious_dd_threshold_pct,
                                          ci_options if with_ci else None)
                            for name, (events, daily, weights, changes) in runs.items()})
    summary = evaluate_period(start, end, with_ci=True)
    periods = {}
    for year in range(start.year, end.year + 1):
        for half, first_month, last_month in [(1, 1, 6), (2, 7, 12)]:
            begin = max(start, pd.Timestamp(year, first_month, 1))
            finish = min(end, pd.Timestamp(year, last_month, 1) + pd.offsets.MonthEnd(0))
            if begin <= finish:
                periods[f"{year} H{half}"] = evaluate_period(begin, finish)
    period_results = pd.concat(periods, names=["Period", "Method"])
    evaluation_equity = pd.concat({name: _equity_path(events, initial_capital, start, end)
                                   for name, (events, _, _, _) in runs.items()}, axis=1)
    evaluation_return = (evaluation_equity / summary["Start_USDT"] - 1) * 100
    evaluation_drawdown = (evaluation_equity / evaluation_equity.cummax() - 1) * 100
    figure = axes = None
    if plot:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        for name in evaluation_equity:
            axes[0].step(evaluation_return.index, evaluation_return[name], where="post", label=name)
            axes[1].step(evaluation_drawdown.index, evaluation_drawdown[name], where="post", label=name)
        axes[0].set(title=f"{method}: {scope}, {start:%Y-%m-%d} — {end:%Y-%m-%d}", ylabel="Доходность, %")
        axes[1].set(ylabel="Просадка капитала, %", xlabel="Дата закрытия сделки")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        if show:
            plt.show()
    trade_result = trades.copy(deep=True)
    trade_result["control_multiplier"] = multiplier
    trade_result["controlled_profit_abs"] = controlled_values
    return {
        "evaluation_pnl": evaluation_return, "evaluation_return_pct": evaluation_return,
        "evaluation_drawdown_pct": evaluation_drawdown, "evaluation_equity": evaluation_equity,
        "summary": summary, "period_results": period_results,
        "thresholds": pd.concat(threshold_tables), "trades": trade_result,
        "modes": modes, "figure": figure, "axes": axes,
        "settings": {"method": method, "scope": scope, "hyperparameters": params,
                     "portfolio_hyperparameters": portfolio_params,
                     "evaluation_start": start, "evaluation_end": end,
                     "calibration_end": start - pd.Timedelta(days=1),
                     "initial_capital": initial_capital, "signal_lag_days": 1,
                     "combined_rule": "minimum", "daily_timezone": "UTC",
                     "individual_signal_unit": "pp (profit_eq_pc)",
                     "portfolio_signal_unit": "pct (cash equity)",
                     "return_unit": "percent of OOS opening balance",
                     "ci_confidence": ci_confidence if compute_ci else None,
                     "period_frequency": "calendar half-year; no control reset",
                     "limitations": "realized PnL only; fixed source volumes; no capital flows or reallocation"},
    }
