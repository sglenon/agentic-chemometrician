# Acceptance evaluation

The project acceptance layer checks manifest and plan hash equality, declared
preparation grouping, evidence locality, conservative mixture closure, and
explicit LOD-design abstention.

Golden acceptance requires that invalid `%T`, missing preparation groups,
group-equals-class leakage, negative or non-closing mixtures, incompatible
reference physical states, non-estimable LOD designs, hash tampering, and
cross-run artifacts are each blocked or explicitly abstained from. A result is
ready only when all applicable scenarios pass and no unsupported claims are
reported.

These checks do not prove chemical identity, purity, method validation,
LOD/LOQ, or generalization merely from similarity or model scores. They count
declared manual transformations but cannot detect undocumented edits. These
are deterministic engineering acceptance criteria; no usability study has
been performed.
