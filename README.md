# model-selection

Репозиторий покрывает упражнения `ex00`-`ex04` и следует структуре задания: окружение и библиотеки, `KFold`, `cross_validate`, `GridSearchCV`, а также `validation curve` и `learning curve`.

## Что выполнено

- `ex00` - минимальная проверка окружения: версия Python и доступность библиотек `jupyter`, `numpy`, `pandas`, `matplotlib`, `sklearn`
- `ex01` - 5-fold разбиение через `KFold` с выводом `TRAIN/TEST` индексов
- `ex02` - `cross_validate` для `LinearRegression` на California Housing с выводом score-массива, среднего и стандартного отклонения
- `ex03` - `GridSearchCV` для `RandomForestRegressor` с `scoring="neg_mean_squared_error"`, выводом `best_score_`, `best_params_`, `cv_results_` и score на test set
- `ex04` - построение `validation curve` и `learning curve` для `RandomForestClassifier`

## Структура проекта

```text
model-selection/
├── ex00/
│   └── main.py
├── ex01/
│   └── main.py
├── ex02/
│   └── main.py
├── ex03/
│   └── main.py
├── ex04/
│   └── main.py
└── .gitignore
├── requirements.txt
└── README.md
```

## Требования

- Python 3.9+
- `jupyter`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Как запускать

### Подготовка окружения

```bash
python -m venv .venv
```

### Активация окружения в Windows (Git Bash)
```bash
source .venv/Scripts/activate
```

### Активация окружения в Linux/macOS
```bash
source .venv/bin/activate
```

### Установка зависимостей
```bash
python -m pip install -r requirements.txt
```

### Запуск проверки окружения

```bash
python ex00/main.py
```

### `KFold`

```bash
python ex01/main.py
```

### `cross validation`

```bash
python ex02/main.py
```

### `GridSearchCV`

```bash
python ex03/main.py
```

### Построить графики `validation curve` и `learning curve`

```bash
python ex04/main.py
```

## Что делает каждое упражнение

### ex00

Проверяет:
- версию Python
- успешный импорт обязательных библиотек

Подходит под минимальную проверку из аудита.

### ex01

Создаёт массивы:

```python
X = np.array(np.arange(1, 21).reshape(10, -1))
y = np.array(np.arange(1, 11))
```

Затем выполняет `KFold(n_splits=5)` и печатает индексы train/test для каждого фолда.

### ex02

Использует датасет California Housing, делит его на train/test и собирает pipeline:

```python
SimpleImputer(strategy="median")
StandardScaler()
LinearRegression()
```

После этого запускает `cross_validate(..., cv=10, scoring="r2")` на train-части и печатает:
- scores по validation folds
- среднее значение
- стандартное отклонение

### ex03

Использует California Housing и запускает `GridSearchCV` для `RandomForestRegressor`.

Параметры поиска:

```python
{
    "n_estimators": [10, 50, 75],
    "max_depth": [4, 7, 10],
}
```

Scoring:

```python
"neg_mean_squared_error"
```

Важно: в `scikit-learn` для этой метрики возвращается отрицательное значение score, потому что поиск лучшей модели в `GridSearchCV` работает по правилу "больше = лучше". Поэтому `best_score_` и `gridsearch.score(...)` здесь ожидаемо отрицательные, хотя сам `MSE` как величина неотрицателен.

### ex04

Состоит из двух частей:

1. `validation curve` для `RandomForestClassifier` по параметру `max_depth`
2. `learning curve` для `RandomForestClassifier(max_depth=12)`

Скрипт открывает окна `matplotlib` с графиками. Если окружение без GUI, может понадобиться headless backend, например:

```bash
MPLBACKEND=Agg python ex04/main.py
```

## TOC

- [Что выполнено](#что-выполнено)
- [Структура проекта](#структура-проекта)
- [Требования](#требования)
- [Как запускать](#как-запускать)
- [Что делает каждое упражнение](#что-делает-каждое-упражнение)


## Автор
- Nazar Yestayev (@nyestaye / @legion2440)