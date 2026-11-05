# Использование drawdown control

Файл `notebooks/drawdown_control.py` содержит функцию `evaluate_drawdown_control` для проверки порогового контроля просадок на данных закрытых сделок. Функция отделяет историю расчёта порогов от периода оценки, сравнивает контролируемый вариант с Baseline и не изменяет переданный DataFrame.

## Подготовка окружения

Из корня репозитория создайте или обновите Conda-окружение:

```bash
conda env create -f environment.yml
conda activate trading-risk-analysis
```

Если окружение уже существует:

```bash
conda env update -f environment.yml --prune
```

Запускайте JupyterLab из каталога `notebooks`, чтобы модуль импортировался без изменения `PYTHONPATH`:

```bash
cd notebooks
jupyter lab
```

В ноутбуке импортируйте функцию:

```python
from drawdown_control import evaluate_drawdown_control
```

## Требования к данным

Аргумент `trades` должен быть непустым `pandas.DataFrame` с закрытыми сделками.

Для любого режима нужны столбцы:

| Столбец | Содержание |
|---|---|
| `entry_date` | Дата и время входа |
| `exit_date` | Дата и время закрытия |
| `profit_abs` | Денежный PnL закрытой сделки |

Для `scope="individual"` и `scope="both"` дополнительно нужны:

| Столбец | Содержание |
|---|---|
| `strategy_name` | Уникальный строковый идентификатор стратегии |
| `profit_eq_pc` | Результат сделки в процентных пунктах капитала |

Если `initial_capital` не передан явно, нужен столбец `start_equity`. Значения начального капитала у сделок с самым ранним временем входа должны совпадать.

Даты преобразуются в UTC, после чего временная зона удаляется. `exit_date` не может предшествовать `entry_date`. Для расчёта порогов обязательно должна существовать история до `evaluation_start`.

## Базовый пример

```python
import pandas as pd
from drawdown_control import evaluate_drawdown_control

trades = pd.read_csv(
    "../data/deals_all_opt_2026-09-11__risk_uniform.csv",
    parse_dates=["entry_date", "exit_date"],
)

result = evaluate_drawdown_control(
    trades=trades,
    method="CDaR",
    scope="individual",
    evaluation_start="2025-01-01",
    evaluation_end="2026-09-11",
    q=0.95,
    recovery_fraction=0.5,
    reduced_multiplier=0.0,
)
```

История строго до `evaluation_start` используется для расчёта порогов. Начиная с этой даты функция выполняет out-of-sample симуляцию. Сигнал применяется с задержкой в один календарный день: сделки текущего дня используют только известные ранее закрытия.

## Методы расчёта порога

### maxDD

```python
result = evaluate_drawdown_control(
    trades,
    method="maxDD",
    scope="individual",
    evaluation_start="2025-01-01",
    k=0.8,
)
```

Порог равен `k × историческая максимальная просадка`. Значение `method="MDD"` является допустимым псевдонимом.

### DaR

```python
result = evaluate_drawdown_control(
    trades,
    method="DaR",
    scope="individual",
    evaluation_start="2025-01-01",
    q=0.95,
)
```

Порог равен квантилю `q` исторического распределения дневной глубины просадки.

### CDaR

```python
result = evaluate_drawdown_control(
    trades,
    method="CDaR",
    scope="individual",
    evaluation_start="2025-01-01",
    q=0.95,
)
```

Порог равен средней глубине исторических просадок, находящихся не ниже DaR с квантилем `q`.

## Область применения контроля

### Отдельные стратегии

```python
result = evaluate_drawdown_control(
    trades,
    method="CDaR",
    scope="individual",
    evaluation_start="2025-01-01",
)
```

Для каждой стратегии рассчитываются собственные порог и режим ограничения. Сигнал строится по виртуальной накопленной `profit_eq_pc`, а денежные результаты портфеля — по `profit_abs`.

### Портфель целиком

```python
result = evaluate_drawdown_control(
    trades,
    method="CDaR",
    scope="portfolio",
    evaluation_start="2025-01-01",
)
```

Портфель рассматривается как единая стратегия. Порог рассчитывается по процентной просадке общей денежной кривой.

### Стратегии и портфель одновременно

```python
result = evaluate_drawdown_control(
    trades,
    method="CDaR",
    scope="both",
    evaluation_start="2025-01-01",
    q=0.95,
    recovery_fraction=0.5,
    reduced_multiplier=0.5,
    portfolio_hyperparameters={
        "q": 0.90,
        "recovery_fraction": 0.7,
        "reduced_multiplier": 0.0,
    },
)
```

Индивидуальный и портфельный множители объединяются правилом минимума. Например, при индивидуальном множителе `0.5` и портфельном `0.0` итоговый множитель равен `0.0`. Параметры, отсутствующие в `portfolio_hyperparameters`, наследуются из основных аргументов.

## Основные гиперпараметры

| Параметр | Допустимые значения | Смысл |
|---|---|---|
| `q` | `0 < q < 1` | Квантиль для DaR/CDaR |
| `k` | `k > 0` | Коэффициент исторической MDD |
| `recovery_fraction` | `0 < value <= 1` | Доля пороговой просадки, которую нужно восстановить |
| `reduced_multiplier` | `0 <= value <= 1` | Размер новых позиций во время ограничения |

Глубина возврата рассчитывается как:

```text
(1 - recovery_fraction) × threshold
```

Поэтому `recovery_fraction=0.5` означает возврат после восстановления половины пороговой просадки, а `1.0` — только после полного восстановления к пику.

`reduced_multiplier=0.0` пропускает новые входы, `0.5` открывает их половинным объёмом. Уже открытые позиции не пересчитываются задним числом.

## Дополнительные параметры

| Параметр | Значение по умолчанию | Назначение |
|---|---:|---|
| `initial_capital` | из `start_equity` | Начальный денежный капитал |
| `evaluation_end` | последняя дата данных | Конец периода оценки |
| `on_insufficient_history` | `"raise"` | Ошибка или отсутствие ограничения при недостаточной истории |
| `serious_dd_threshold_pct` | `3.0` | Порог серьёзной просадки в процентах |
| `compute_ci` | `True` | Рассчитывать интервалы DaR/CDaR |
| `ci_resamples` | `2000` | Число bootstrap-выборок |
| `ci_block_days` | `30` | Длина блока в календарных днях |
| `ci_confidence` | `0.95` | Уровень доверия |
| `random_state` | `42` | Воспроизводимость bootstrap |
| `plot` | `True` | Создавать сравнительный график |
| `show` | `True` | Сразу отображать график |

Если отдельная стратегия не имеет положительного порога до даты оценки, режим `on_insufficient_history="raise"` остановит расчёт. Значение `"unrestricted"` оставит такую стратегию без ограничения и выдаст предупреждение.

## Результат функции

Функция возвращает словарь со следующими ключами:

| Ключ | Содержание |
|---|---|
| `evaluation_return_pct` | Доходность Baseline и контролируемого варианта в процентах начального OOS-баланса |
| `evaluation_pnl` | Псевдоним `evaluation_return_pct` для совместимости |
| `evaluation_drawdown_pct` | Процентная просадка обеих кривых |
| `evaluation_equity` | Денежные кривые капитала |
| `summary` | Итоговые метрики всего периода оценки |
| `period_results` | Метрики по календарным полугодиям |
| `thresholds` | Рассчитанные пороги и глубины восстановления |
| `trades` | Копия сделок с `control_multiplier` и `controlled_profit_abs` |
| `modes` | Индивидуальные, портфельные и итоговые режимы по дням |
| `figure`, `axes` | Объекты Matplotlib либо `None` |
| `settings` | Фактически использованные настройки и ограничения модели |

Основные таблицы можно просмотреть так:

```python
display(result["summary"])
display(result["period_results"])
display(result["thresholds"])
```

Отфильтровать сделки, размер которых был изменён:

```python
controlled = result["trades"]
controlled.loc[controlled["control_multiplier"] < 1, [
    "strategy_name",
    "entry_date",
    "exit_date",
    "profit_abs",
    "control_multiplier",
    "controlled_profit_abs",
]]
```

Сохранить график без автоматического отображения:

```python
result = evaluate_drawdown_control(
    trades,
    method="CDaR",
    scope="both",
    evaluation_start="2025-01-01",
    plot=True,
    show=False,
)
result["figure"].savefig("drawdown_control_comparison.png", dpi=160, bbox_inches="tight")
```

## Интерпретация метрик

`summary` содержит PnL, доходность, CAGR, MDD, DaR/CDaR, Calmar, Recovery Factor, Sharpe, Sortino, длительность просадки, время под водой, количество и длительность серьёзных просадок, долю уменьшенных/пропущенных сделок и количество переключений режима.

Контролируемый вариант следует сравнивать с Baseline сразу по нескольким критериям. Снижение MDD само по себе недостаточно, если оно сопровождается непропорциональной потерей PnL, ухудшением Calmar или чрезмерным количеством переключений.

## Ограничения модели

- Используются только закрытые позиции и реализованный PnL.
- Контроль применяется к размеру новых входов; уже открытые позиции не изменяются.
- Исходные объёмы фиксированы, высвободившийся капитал не перераспределяется.
- Не моделируются проскальзывание, funding, ликвидность и задержки исполнения.
- Порог рассчитывается только по истории до `evaluation_start`, но корректность out-of-sample вывода всё равно зависит от того, подбирались ли гиперпараметры на итоговом периоде.
- Интервалы DaR/CDaR рассчитываются блочным bootstrap по дневной кривой просадки и характеризуют неопределённость метрик, а не гарантированный будущий риск.
