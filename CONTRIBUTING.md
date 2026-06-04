# Contributing

Thank you for helping make MedMLOps-Lab more reproducible and useful.

## Development Setup

```bash
make setup
make test
```

## Commit Style

Use Conventional Commits:

- `feat: add validation report`
- `fix: preserve readmission target domain`
- `docs: explain DVC remote setup`
- `test: cover invalid admission type`

## Pull Request Rules

- Keep PRs scoped to one behavior or documentation change.
- Include tests for code changes, or explain why tests are not applicable.
- Use DVC for generated data, model, and metric artifacts.
- Do not commit raw clinical, proprietary, or non-redistributable data.
- Update docs when changing public commands, configs, or behavior.

## Data And Model Governance

The default dataset is redistributable under CC BY 4.0. New datasets require a
documented source, license, citation, intended use, prohibited use, and data
contract before they can enter the pipeline.
