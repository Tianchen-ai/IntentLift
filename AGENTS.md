# Contribution guidance

- Preserve the separation between recovered algorithm, source checking, and target implementation. Target emitters must not select algorithms by kernel name or read source / Replay programs.
- Distinguish build success, example execution, fresh Gate acceptance, and measured performance. Saved examples and paper tables do not count as fresh recovery results.
- Do not add or restore test directories or substitute mock validation for real kernel evidence. Use the relevant compiler/example workflow for packaging changes and real Gate artifacts for recovery claims.
- Preserve existing wire-format compatibility tags; do not introduce version-label churn.
- Keep credentials, raw model transcripts, local paths, private paper material, and generated build products out of commits.
- Maintain documentation of interfaces and reproducible workflows rather than per-change status diaries. Use an external task record for ongoing work.
- Preserve third-party licenses and source provenance.
