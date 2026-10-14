# Restricted access, Qwen3-1.7B

| run | overall | tracked updated | colour kept (absent) | at a switch | no memory (chunks 1-5) | on tracked record | on "nothing new" when absent |
|---|---|---|---|---|---|---|---|
| sel-const_l0 | 0.281 | 0.332 | 0.112 | 0.292 | 0.266 | 0.255 | 0.064 |
| sel-oracle_l0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.751 | 1.000 | 1.000 |
| sel-state_l0 | 0.653 | 0.808 | 0.137 | 0.357 | 0.320 | 0.796 | 0.065 |
| sel-state_l0_s1 | 0.763 | 0.936 | 0.187 | 0.318 | 0.305 | 0.935 | 0.129 |

| arm | seeds | overall | tracked updated | colour kept | at a switch |
|---|---|---|---|---|---|
| sel-const_l0 | 1 | 0.281 ± 0.000 | 0.332 ± 0.000 | 0.112 ± 0.000 | 0.292 ± 0.000 |
| sel-oracle_l0 | 1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| sel-state_l0 | 2 | 0.708 ± 0.055 | 0.872 ± 0.064 | 0.162 ± 0.025 | 0.338 ± 0.019 |

Seed_s0: state minus constant query, per stream (paired, 95 % bootstrap over streams): overall +0.372 [+0.348, +0.397]; chunks 1-5 only (no instruction in view) +0.315 [+0.289, +0.342]; chunks where the tracked animal is updated +0.481 [+0.454, +0.511].

| run | own: answers A / B | sham: A / B | query from B: A / B | replay chunks 1-t: follows B / A | after restoring: follows A |
|---|---|---|---|---|---|
| sel-const_l0 | 0.273 / 0.273 | – | – | – | – |
| sel-oracle_l0 | 1.000 / 0.000 | – | – | – | – |
| sel-state_l0 | 0.734 / 0.133 | 0.734 / 0.133 | 0.113 / 0.762 | 0.477 / 0.189 | 0.711 |
| sel-state_l0_s1 | 0.992 / 0.000 | 0.992 / 0.000 | 0.000 / 0.996 | 0.598 / 0.143 | 0.994 |

| run | tracked animal from the state | its colour from the state | tracked animal from the query | chance |
|---|---|---|---|---|
| sel-const_l0 | 0.673 | 0.294 | 0.105 | 0.100 |
| sel-oracle_l0 | 0.119 | 1.000 | – | 0.100 |
| sel-state_l0 | 0.744 | 0.698 | 0.695 | 0.100 |
| sel-state_l0_s1 | 0.973 | 0.794 | 0.941 | 0.100 |

| run | "took in something new" from the state (balanced acc.) | selection confidence from the state (R²) | self-model keeps the focus (forecast vs next state; persistence) |
|---|---|---|---|
| sel-const_l0 | 0.861 | nan | 0.324; 0.604 |
| sel-oracle_l0 | 1.000 | nan | – |
| sel-state_l0 | 0.902 | 0.059 | 0.395; 0.706 |
| sel-state_l0_s1 | 0.851 | -0.014 | 0.829; 0.926 |

| run | cases where own choice ≠ what happened | decoder of own choice right there | decoder of what happened right there |
|---|---|---|---|
| sel-const_l0 | 832 (27.1%) | 0.875 | 0.731 |
| sel-oracle_l0 | 0 | – | – |
| sel-state_l0 | 670 (21.8%) | 0.979 | 0.709 |
| sel-state_l0_s1 | 628 (20.4%) | 0.889 | 0.742 |
