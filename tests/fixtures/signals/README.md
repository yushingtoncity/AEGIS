Recorded-shape fixtures for the signal sources. Synthetic values, shaped
after the provider's documentation (read 2026-10-30):

- `apewisdom_page*.json`: https://apewisdom.io/api/ (fields `count`, `pages`,
  `current_page`, `results[]` with `rank`, `ticker`, `name`, `mentions`,
  `upvotes`, `rank_24h_ago`, `mentions_24h_ago`). Page 1 mixes numbers and
  numeric strings and carries two malformed rows on purpose.

Alpaca News signals reuse `tests/fixtures/news.json`.
