# RUIN v0.1 Command Examples

Primary v0.1 simulation command:

```text
ruin simulate --config examples/urban_ruin_shift.yaml --visualize --frames 300
```

Risk-only Monte Carlo command shape:

```text
ruin risk --config examples/urban_ruin_shift.yaml --paths 1000 --confidence 0.95
```

The first command should run one visible Probability Square destiny. The second command should estimate finite-time ruin probability and tail-loss metrics across many destinies.
