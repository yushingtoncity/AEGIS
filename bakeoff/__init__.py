"""Offline model bakeoff for the thesis stage.

Replays frozen, reconstructed thesis inputs through several models and
records quality and cost in a separate SQLite file. Nothing here is part of
the live cycle: no module under ``aegis`` imports this package, and this
package never imports the order path (the execution package, the policy
dispatcher, the brain cycle or the CLIs). ``tests/test_bakeoff_architecture.py``
pins both directions.

Modules:

- ``config``    the bakeoff's own settings (``bakeoff/bakeoff.yaml``)
- ``dataset``   rebuild thesis inputs from the audit DB, freeze them as JSONL
- ``providers`` one small interface per model provider (Anthropic, Gemini)
- ``standin``   a sized stand in proposal and an offline policy dry run
- ``critic``    the shadow critic: a logged opinion, never a gate
- ``results``   the results SQLite schema
- ``runner``    every input through every contestant, spend cap, retries
- ``report``    the markdown report and the blind grading CSV
"""
