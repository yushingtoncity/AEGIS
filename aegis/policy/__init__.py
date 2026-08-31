"""Phase 5 — deterministic policy engine (stub).

Will validate TradeProposals against the risk_limits block of config.yaml
(max_position_pct, max_open_positions, daily_loss_limit_pct, no_trade_list)
plus account and market-state checks — no model calls, fully deterministic
and unit-testable. The single gatekeeper: only this package may pass
approved orders to aegis.execution.
"""
