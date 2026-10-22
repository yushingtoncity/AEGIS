"""Per-ticker forecasts: the live inputs the forecast stage is shown, and the
read-only scorer that grades what it wrote.

``inputs`` reads market data through ``aegis.data`` (the brain may not) and
hands the forecast stage (``aegis.brain.forecast``) a typed
``ForecastInputs``. ``events`` defines each forecast's outcome exactly, and
``score`` grades the forecasts against adjusted closes with log loss, Brier
scores, no-skill baselines and the pre-registered primary test. Nothing in
this package reads or writes anything that can place an order.
"""
