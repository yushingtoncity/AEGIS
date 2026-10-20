# AEGIS agent instructions

Before any AEGIS work, read `docs/STATUS.md` from the current GitHub `main`
branch as the first status source. If using a local checkout, verify its
revision against GitHub `main` before relying on its copy. Check whether
`main` has changed since the status file's verified commit; re-check any
relevant code or config changes before treating the status as current.

Code and config at the relevant commit outrank `docs/STATUS.md`; the status
file outranks PR descriptions, assistant memory, chat summaries, and older
documentation. If the file cannot be read, say so and do not substitute
remembered status as verified current information.

Then read and follow `CLAUDE.md` for repository safety and contribution rules.
Work on branches and pull requests only; the owner reviews and merges.
Status changes go through the same reviewed PR workflow described in
`docs/STATUS.md`. Do not enable broker execution or notifications, weaken
policy gates, or use UI/UX skills as part of status review.

Read status on demand at the start of each task. This instruction does not
create background monitoring, automatic updates, or permission to run trading
operations.
