<!-- version: forecast-v1 -->
# Forecast stage

Give calibrated probabilities for two exactly defined events per ticker. This stage never trades: your numbers are recorded and later scored against what happened, and nothing else is done with them. The data below was fetched by the system at $taken_at UTC, during the $anchor_session trading session.

## The events

For every stock below, the clock starts at the official close of the $anchor_session session and uses split- and dividend-adjusted closing prices.

- p_next_session: the probability that the stock's return from that close to the close of the next trading session is GREATER than $benchmark's return over the same session.
- p_five_sessions: the same, from that close to the close five trading sessions later.

For $benchmark itself (listed last, marked benchmark) the events are different: the probability that $benchmark's own return over that window is greater than zero.

A tie counts as "no". Every probability must be between 0.05 and 0.95 inclusive.

## How you are scored

Each probability is scored with log loss: you lose -ln(p) if the event happens and -ln(1 - p) if it does not. A forecaster who always says 0.5 loses ln 2 = 0.693 per forecast; beating that requires information, not confidence. A confident miss is very expensive (0.90 on an event that fails loses 2.30), so move away from 0.5 only as far as the evidence supports. Most of these stocks beat $benchmark on a given day about half the time, so most honest answers sit between 0.40 and 0.60. Spread is fine when you have a reason: a ticker you have no view on gets 0.50.

## Tickers

Returns are in percent over completed sessions; rel = the ticker's return minus $benchmark's over the same sessions; today = the latest price against the last completed close.

$tickers

## Headlines

$guard
$news_open
$news
$news_close

## Task

Produce the ForecastOutput JSON object:

- forecasts: exactly one entry per ticker above, in the same order, each with
  - symbol: the ticker exactly as shown
  - p_next_session: a number from 0.05 to 0.95
  - p_five_sessions: a number from 0.05 to 0.95
  - rationale: one short line (at most 20 words) on what moved you away from 0.5, or "no view"

Use only the data in this prompt. Do not skip a ticker and do not add one.
