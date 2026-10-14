# Open access (section 5.5)
## open_z32 — Qwen/Qwen2.5-0.5B-Instruct (state of 32 numbers; written at block 8, read at block 16)

| overall | tracked animal updated | colour kept (absent) | instruction-free chunks | same, no memory | per stream: with minus without memory (instruction-free) |
|---|---|---|---|---|---|
| 0.442 | 0.514 | 0.212 | 0.291 | 0.261 | +0.030 [+0.008, +0.051] |

By chunk: 1.00, 0.29, 0.35, 0.36, 0.36, 0.29 (no memory: 1.00, 0.24, 0.33, 0.33, 0.34, 0.27)

What the state holds after a chunk (cross-validated decoding, whole streams held out; chance about 0.10):

| chunks | decoder | new colour of the tracked animal | new colours of the other animals updated in the same chunk |
|---|---|---|---|
| 0 (instruction in view) | linear | 1.000 (n 512) | 0.115 (n 1536; per animal [0.08, 0.127, 0.107, 0.136, 0.111, 0.111, 0.119, 0.115, 0.086, 0.159]) |
| 0 (instruction in view) | nonlinear | 0.988 (n 512) | 0.096 (n 1536; per animal [0.053, 0.076, 0.093, 0.107, 0.111, 0.104, 0.113, 0.101, 0.098, 0.106]) |
| 1-5 | linear | 0.425 (n 1829) | 0.349 (n 8411; per animal [0.39, 0.337, 0.366, 0.306, 0.318, 0.356, 0.391, 0.292, 0.365, 0.369]) |
| 1-5 | nonlinear | 0.382 (n 1829) | 0.319 (n 8411; per animal [0.366, 0.304, 0.339, 0.271, 0.314, 0.333, 0.378, 0.262, 0.304, 0.316]) |

Focus edits on twin streams (edit at chunk 2, 3 or 4; 768 pairs; 95 % intervals):

| edit | A's new colour | B's new colour | A's previous colour | next chunk: A / B |
|---|---|---|---|---|
| no edit | 0.286 [0.26, 0.32] | 0.332 [0.30, 0.37] | 0.118 [0.10, 0.14] | 0.294 / 0.228 |
| focus A → B before the chunk is read | 0.283 [0.25, 0.32] | 0.337 [0.30, 0.37] | 0.118 [0.10, 0.14] | 0.294 / 0.228 |
| same edit after the chunk is read (question only) | 0.294 [0.26, 0.33] | 0.339 [0.31, 0.37] | 0.120 [0.10, 0.14] | 0.294 / 0.228 |
| half strength | 0.298 [0.27, 0.33] | 0.326 [0.29, 0.36] | 0.115 [0.09, 0.14] | 0.299 / 0.234 |
| random direction, same size | 0.281 [0.25, 0.31] | 0.333 [0.30, 0.37] | 0.117 [0.10, 0.14] | 0.297 / 0.230 |
| toward an animal absent from the chunk | 0.292 [0.26, 0.32] | 0.326 [0.29, 0.36] | 0.117 [0.10, 0.14] | 0.296 / 0.224 |

The frozen model's own attention from the chunk's last token while it reads chunk 3 (mean over heads and over layers 9-16; twin A streams):

| condition | on A's record | on B's record | on each other record | on the placeholders |
|---|---|---|---|---|
| trained state | 0.1299 | 0.1213 | 0.1265 | 0.1403 |
| focus edit A → B | 0.1299 | 0.1213 | 0.1265 | 0.1405 |
| no memory (initial state) | 0.1305 | 0.1219 | 0.1279 | 0.1789 |
| nothing written | 0.1457 | 0.1355 | 0.1429 | 0.0110 |

Per layer, A / B (trained state → after the edit): 9: 0.125/0.118 → 0.125/0.118; 10: 0.142/0.128 → 0.142/0.128; 11: 0.059/0.061 → 0.059/0.061; 12: 0.180/0.163 → 0.180/0.163; 13: 0.115/0.105 → 0.115/0.105; 14: 0.176/0.168 → 0.176/0.168; 15: 0.141/0.130 → 0.141/0.129; 16: 0.102/0.099 → 0.102/0.099

Chunk 0, instruction in view (A's stream says A, its twin says B; both records are in the chunk): attention from the last token

| instruction | write | on A's record | on B's record | on each other record | on the placeholders |
|---|---|---|---|---|---|
| A | learned write | 0.1787 | 0.0953 | 0.1044 | 0.1722 |
| A | nothing written | 0.1391 | 0.1335 | 0.1470 | 0.0101 |
| B | learned write | 0.1050 | 0.1678 | 0.1055 | 0.1713 |
| B | nothing written | 0.1469 | 0.1253 | 0.1472 | 0.0100 |

## open_z32 — Qwen/Qwen2.5-3B-Instruct (state of 32 numbers; written at block 12, read at block 24)

| overall | tracked animal updated | colour kept (absent) | instruction-free chunks | same, no memory | per stream: with minus without memory (instruction-free) |
|---|---|---|---|---|---|
| 0.428 | 0.489 | 0.234 | 0.286 | 0.259 | +0.025 [+0.003, +0.047] |

By chunk: 1.00, 0.29, 0.34, 0.32, 0.34, 0.29 (no memory: 1.00, 0.23, 0.33, 0.33, 0.34, 0.27)

What the state holds after a chunk (cross-validated decoding, whole streams held out; chance about 0.10):

| chunks | decoder | new colour of the tracked animal | new colours of the other animals updated in the same chunk |
|---|---|---|---|
| 0 (instruction in view) | linear | 1.000 (n 512) | 0.109 (n 1536; per animal [0.14, 0.083, 0.107, 0.093, 0.135, 0.052, 0.081, 0.122, 0.109, 0.159]) |
| 0 (instruction in view) | nonlinear | 0.973 (n 512) | 0.104 (n 1536; per animal [0.113, 0.115, 0.127, 0.129, 0.105, 0.089, 0.069, 0.095, 0.098, 0.106]) |
| 1-5 | linear | 0.382 (n 1829) | 0.334 (n 8411; per animal [0.307, 0.31, 0.31, 0.345, 0.338, 0.337, 0.343, 0.374, 0.333, 0.346]) |
| 1-5 | nonlinear | 0.356 (n 1829) | 0.307 (n 8411; per animal [0.275, 0.271, 0.304, 0.302, 0.333, 0.341, 0.317, 0.334, 0.297, 0.296]) |

Focus edits on twin streams (edit at chunk 2, 3 or 4; 768 pairs; 95 % intervals):

| edit | A's new colour | B's new colour | A's previous colour | next chunk: A / B |
|---|---|---|---|---|
| no edit | 0.310 [0.28, 0.34] | 0.314 [0.28, 0.35] | 0.095 [0.08, 0.12] | 0.259 / 0.223 |
| focus A → B before the chunk is read | 0.311 [0.28, 0.34] | 0.316 [0.28, 0.35] | 0.098 [0.08, 0.12] | 0.263 / 0.229 |
| same edit after the chunk is read (question only) | 0.307 [0.28, 0.34] | 0.335 [0.30, 0.37] | 0.100 [0.08, 0.12] | 0.259 / 0.223 |
| half strength | 0.314 [0.28, 0.35] | 0.315 [0.28, 0.35] | 0.105 [0.09, 0.13] | 0.257 / 0.240 |
| random direction, same size | 0.314 [0.28, 0.35] | 0.322 [0.29, 0.36] | 0.092 [0.07, 0.12] | 0.254 / 0.232 |
| toward an animal absent from the chunk | 0.307 [0.28, 0.34] | 0.319 [0.29, 0.35] | 0.099 [0.08, 0.12] | 0.264 / 0.224 |

The frozen model's own attention from the chunk's last token while it reads chunk 3 (mean over heads and over layers 13-24; twin A streams):

| condition | on A's record | on B's record | on each other record | on the placeholders |
|---|---|---|---|---|
| trained state | 0.1385 | 0.1317 | 0.1368 | 0.1380 |
| focus edit A → B | 0.1385 | 0.1318 | 0.1368 | 0.1378 |
| no memory (initial state) | 0.1296 | 0.1219 | 0.1280 | 0.1720 |
| nothing written | 0.1500 | 0.1403 | 0.1467 | 0.0206 |

Per layer, A / B (trained state → after the edit): 13: 0.087/0.079 → 0.087/0.079; 14: 0.150/0.135 → 0.150/0.135; 15: 0.156/0.145 → 0.156/0.145; 16: 0.136/0.120 → 0.136/0.120; 17: 0.094/0.092 → 0.094/0.092; 18: 0.169/0.153 → 0.169/0.153; 19: 0.110/0.107 → 0.111/0.107; 20: 0.128/0.127 → 0.128/0.127; 21: 0.148/0.157 → 0.147/0.157; 22: 0.164/0.165 → 0.164/0.165; 23: 0.136/0.129 → 0.136/0.129; 24: 0.185/0.172 → 0.185/0.172

Chunk 0, instruction in view (A's stream says A, its twin says B; both records are in the chunk): attention from the last token

| instruction | write | on A's record | on B's record | on each other record | on the placeholders |
|---|---|---|---|---|---|
| A | learned write | 0.1790 | 0.0961 | 0.1047 | 0.1729 |
| A | nothing written | 0.1541 | 0.1335 | 0.1489 | 0.0184 |
| B | learned write | 0.1077 | 0.1678 | 0.1051 | 0.1721 |
| B | nothing written | 0.1515 | 0.1356 | 0.1488 | 0.0185 |

