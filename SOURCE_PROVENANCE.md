# Source provenance

The candidate uses five provenance classes:

- Research-added: original study code or documentation.
- Based-on-DeepfakeBench: a modified copy of upstream source.
- Third-party-dependency: external software obtained separately by users.
- Third-party-reference: metadata or citation pointing to external work,
  without copying its source code.
- Generated-example-result: text summaries or audit records generated from the
  frozen experiment or release checks.

This candidate contains no file in the Based-on-DeepfakeBench class. The
integration helpers were written independently, and the modified upstream test
program is intentionally excluded. DeepfakeBench and the five detector methods
are referenced as external dependencies or citations at recorded versions.

File-level provenance is recorded in `SOURCE_FILES.csv`.
