# Restricted access, Qwen2.5-0.5B-Instruct (every arm; the same 512 held-out streams)

| run | overall | tracked updated | colour kept (absent) | at a switch | no memory (chunks 1-5) | on tracked record | on "nothing new" when absent |
|---|---|---|---|---|---|---|---|
| sel-const_l0 | 0.317 | 0.363 | 0.164 | 0.377 | 0.299 | 0.246 | 0.018 |
| sel-const_l0_s1 | 0.318 | 0.362 | 0.171 | 0.351 | 0.273 | 0.202 | 0.203 |
| sel-fcst_l0.05 | 0.704 | 0.843 | 0.240 | 0.916 | 0.330 | 0.578 | 0.340 |
| sel-instr_l0 | 0.456 | 0.515 | 0.259 | 1.000 | 0.343 | 0.405 | 0.173 |
| sel-oracle_l0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.752 | 1.000 | 1.000 |
| sel-state_l0.05 | 0.992 | 0.999 | 0.967 | 0.981 | 0.204 | 0.862 | 0.759 |
| sel-state_l0 | 0.995 | 1.000 | 0.980 | 1.000 | 0.263 | 0.828 | 0.819 |
| sel-state_l0_s1 | 0.991 | 0.998 | 0.967 | 0.987 | 0.296 | 0.892 | 0.801 |
| sel-state_l0_s2 | 0.987 | 0.998 | 0.948 | 1.000 | 0.169 | 0.876 | 0.825 |

| arm | seeds | overall | tracked updated | colour kept | at a switch |
|---|---|---|---|---|---|
| sel-const_l0 | 2 | 0.318 ± 0.000 | 0.363 ± 0.000 | 0.168 ± 0.004 | 0.364 ± 0.013 |
| sel-fcst_l0.05 | 1 | 0.704 ± 0.000 | 0.843 ± 0.000 | 0.240 ± 0.000 | 0.916 ± 0.000 |
| sel-instr_l0 | 1 | 0.456 ± 0.000 | 0.515 ± 0.000 | 0.259 ± 0.000 | 1.000 ± 0.000 |
| sel-oracle_l0 | 1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| sel-state_l0.05 | 1 | 0.992 ± 0.000 | 0.999 ± 0.000 | 0.967 ± 0.000 | 0.981 ± 0.000 |
| sel-state_l0 | 3 | 0.991 ± 0.003 | 0.999 ± 0.001 | 0.965 ± 0.013 | 0.996 ± 0.006 |

Seed_s0: state minus constant query, per stream (paired, 95 % bootstrap over streams): overall +0.678 [+0.660, +0.695]; instruction-free chunks only (not chunk 0, not a switch) +0.696 [+0.676, +0.715]; chunks where the tracked animal is updated +0.639 [+0.619, +0.659].

Seed_s1: state minus constant query, per stream (paired, 95 % bootstrap over streams): overall +0.673 [+0.650, +0.695]; instruction-free chunks only (not chunk 0, not a switch) +0.690 [+0.665, +0.714]; chunks where the tracked animal is updated +0.635 [+0.610, +0.659].

Seed_s0: state minus instruction-only query (matched control), per stream (paired, 95 % bootstrap over streams): overall +0.539 [+0.523, +0.555]; instruction-free chunks only (not chunk 0, not a switch) +0.688 [+0.668, +0.707]; chunks where the tracked animal is updated +0.474 [+0.457, +0.492].

| run | own: answers A / B | sham: A / B | query from B: A / B | replay chunks 1-t: follows B / A | after restoring: follows A |
|---|---|---|---|---|---|
| sel-const_l0 | 0.273 / 0.316 | – | – | – | – |
| sel-const_l0_s1 | 0.285 / 0.344 | – | – | – | – |
| sel-fcst_l0.05 | 0.754 / 0.117 | 0.754 / 0.117 | 0.059 / 0.809 | 0.508 / 0.173 | 0.799 |
| sel-instr_l0 | 0.273 / 0.309 | 0.273 / 0.309 | 0.273 / 0.309 | 0.219 / 0.314 | 0.361 |
| sel-oracle_l0 | 1.000 / 0.000 | – | – | – | – |
| sel-state_l0.05 | 1.000 / 0.000 | 1.000 / 0.000 | 0.000 / 1.000 | 0.671 / 0.120 | 1.000 |
| sel-state_l0 | 1.000 / 0.000 | 1.000 / 0.000 | 0.000 / 1.000 | 0.671 / 0.118 | 0.997 |
| sel-state_l0_s1 | 1.000 / 0.000 | 1.000 / 0.000 | 0.000 / 1.000 | 0.671 / 0.120 | 0.997 |
| sel-state_l0_s2 | 1.000 / 0.000 | 1.000 / 0.000 | 0.000 / 1.000 | 0.671 / 0.120 | 0.989 |

| run | tracked animal from the state | its colour from the state | tracked animal from the query | chance |
|---|---|---|---|---|
| sel-const_l0 | 0.130 | 0.342 | 0.105 | 0.100 |
| sel-const_l0_s1 | 0.220 | 0.317 | 0.105 | 0.100 |
| sel-fcst_l0.05 | 0.728 | 0.751 | 0.682 | 0.100 |
| sel-instr_l0 | 0.157 | 0.579 | 0.157 | 0.100 |
| sel-oracle_l0 | 0.112 | 0.999 | – | 0.100 |
| sel-state_l0.05 | 1.000 | 0.985 | 0.993 | 0.100 |
| sel-state_l0 | 0.999 | 0.990 | 0.994 | 0.100 |
| sel-state_l0_s1 | 0.999 | 0.975 | 0.993 | 0.100 |
| sel-state_l0_s2 | 1.000 | 0.956 | 0.991 | 0.100 |

| run | "took in something new" from the state (balanced acc.) | selection confidence from the state (R²) | self-model keeps the focus (forecast vs next state; persistence) |
|---|---|---|---|
| sel-const_l0 | – | -0.003 | 0.114; 0.169 |
| sel-const_l0_s1 | – | 0.801 | 0.100; 0.235 |
| sel-fcst_l0.05 | 0.919 | 0.773 | 0.282; 0.707 |
| sel-instr_l0 | – | 0.981 | 0.110; 0.181 |
| sel-oracle_l0 | 1.000 | nan | 0.102; 0.159 |
| sel-state_l0.05 | 0.995 | 0.624 | 0.939; 0.940 |
| sel-state_l0 | 0.988 | 0.768 | 0.940; 0.940 |
| sel-state_l0_s1 | 0.988 | 0.612 | 0.938; 0.940 |
| sel-state_l0_s2 | 0.920 | 0.588 | 0.939; 0.940 |

| run | cases where own choice ≠ what happened | decoder of own choice right there | decoder of what happened right there |
|---|---|---|---|
| sel-const_l0 | 707 | – | – |
| sel-const_l0_s1 | 707 | – | – |
| sel-fcst_l0.05 | 579 (18.8%) | 0.827 | 0.725 |
| sel-instr_l0 | 707 | – | – |
| sel-oracle_l0 | 0 | – | – |
| sel-state_l0.05 | 17 | – | – |
| sel-state_l0 | 3 | – | – |
| sel-state_l0_s1 | 4 | – | – |
| sel-state_l0_s2 | 5 | – | – |
