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

## open_z32_q — Qwen/Qwen2.5-0.5B-Instruct (state of 32 numbers; written at block 8, read at block 16)

| overall | tracked animal updated | colour kept (absent) | instruction-free chunks | same, no memory | per stream: with minus without memory (instruction-free) |
|---|---|---|---|---|---|
| 0.516 | 0.611 | 0.212 | 0.386 | 0.259 | +0.125 [+0.102, +0.148] |

By chunk: 1.00, 0.47, 0.48, 0.42, 0.38, 0.35 (no memory: 1.00, 0.23, 0.33, 0.33, 0.34, 0.27)

What the state holds after a chunk (cross-validated decoding, whole streams held out; chance about 0.10):

| chunks | decoder | new colour of the tracked animal | new colours of the other animals updated in the same chunk |
|---|---|---|---|
| 0 (instruction in view) | linear | 1.000 (n 512) | 0.104 (n 1536; per animal [0.073, 0.127, 0.06, 0.114, 0.088, 0.119, 0.119, 0.095, 0.109, 0.132]) |
| 0 (instruction in view) | nonlinear | 0.990 (n 512) | 0.097 (n 1536; per animal [0.073, 0.121, 0.08, 0.093, 0.129, 0.133, 0.094, 0.081, 0.086, 0.079]) |
| 1-5 | linear | 0.531 (n 1829) | 0.294 (n 8411; per animal [0.3, 0.267, 0.303, 0.363, 0.299, 0.233, 0.303, 0.253, 0.305, 0.315]) |
| 1-5 | nonlinear | 0.504 (n 1829) | 0.278 (n 8411; per animal [0.282, 0.235, 0.289, 0.362, 0.283, 0.214, 0.283, 0.26, 0.295, 0.278]) |

Focus edits on twin streams (edit at chunk 2, 3 or 4; 768 pairs; 95 % intervals):

| edit | A's new colour | B's new colour | A's previous colour | next chunk: A / B |
|---|---|---|---|---|
| no edit | 0.457 [0.42, 0.49] | 0.217 [0.19, 0.25] | 0.105 [0.09, 0.13] | 0.402 / 0.219 |
| focus A → B before the chunk is read | 0.243 [0.21, 0.28] | 0.441 [0.41, 0.48] | 0.129 [0.11, 0.15] | 0.320 / 0.283 |
| same edit after the chunk is read (question only) | 0.469 [0.43, 0.50] | 0.234 [0.21, 0.27] | 0.109 [0.09, 0.13] | 0.402 / 0.219 |
| half strength | 0.376 [0.34, 0.41] | 0.301 [0.27, 0.33] | 0.120 [0.10, 0.14] | 0.365 / 0.242 |
| random direction, same size | 0.426 [0.39, 0.46] | 0.237 [0.21, 0.27] | 0.120 [0.10, 0.14] | 0.397 / 0.214 |
| toward an animal absent from the chunk | 0.299 [0.27, 0.33] | 0.303 [0.27, 0.34] | 0.124 [0.10, 0.15] | 0.303 / 0.234 |

The frozen model's own attention from the chunk's last token while it reads chunk 3 (mean over heads and over layers 9-16; twin A streams):

| condition | on A's record | on B's record | on each other record | on the placeholders |
|---|---|---|---|---|
| trained state | 0.1454 | 0.1362 | 0.1374 | 0.0451 |
| focus edit A → B | 0.1410 | 0.1417 | 0.1389 | 0.0438 |
| no memory (initial state) | 0.1245 | 0.1246 | 0.1257 | 0.0575 |
| nothing written | 0.1489 | 0.1405 | 0.1417 | 0.0214 |

Per layer, A / B (trained state → after the edit): 9: 0.167/0.157 → 0.170/0.158; 10: 0.139/0.126 → 0.137/0.124; 11: 0.069/0.074 → 0.065/0.072; 12: 0.189/0.185 → 0.188/0.186; 13: 0.144/0.141 → 0.151/0.144; 14: 0.186/0.178 → 0.181/0.185; 15: 0.117/0.112 → 0.117/0.111; 16: 0.151/0.116 → 0.120/0.153

Chunk 0, instruction in view (A's stream says A, its twin says B; both records are in the chunk): attention from the last token

| instruction | write | on A's record | on B's record | on each other record | on the placeholders |
|---|---|---|---|---|---|
| A | learned write | 0.1363 | 0.0876 | 0.0930 | 0.0478 |
| A | nothing written | 0.1435 | 0.0949 | 0.1005 | 0.0245 |
| B | learned write | 0.0925 | 0.1294 | 0.0937 | 0.0476 |
| B | nothing written | 0.1005 | 0.1367 | 0.1014 | 0.0243 |

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

## open_z32_more — Qwen/Qwen2.5-3B-Instruct (state of 32 numbers; written at block 12, read at block 24)

| overall | tracked animal updated | colour kept (absent) | instruction-free chunks | same, no memory | per stream: with minus without memory (instruction-free) |
|---|---|---|---|---|---|
| 0.436 | 0.500 | 0.237 | 0.279 | 0.247 | +0.032 [+0.003, +0.062] |

By chunk: 1.00, 0.26, 0.38, 0.32, 0.37, 0.29 (no memory: 1.00, 0.19, 0.36, 0.31, 0.35, 0.27)

What the state holds after a chunk (cross-validated decoding, whole streams held out; chance about 0.10):

| chunks | decoder | new colour of the tracked animal | new colours of the other animals updated in the same chunk |
|---|---|---|---|
| 0 (instruction in view) | linear | 1.000 (n 256) | 0.128 (n 768; per animal [0.145, 0.171, 0.068, 0.083, 0.165, 0.016, 0.133, 0.208, 0.058, 0.205]) |
| 0 (instruction in view) | nonlinear | 0.930 (n 256) | 0.111 (n 768; per animal [0.101, 0.085, 0.096, 0.097, 0.139, 0.129, 0.1, 0.125, 0.105, 0.133]) |
| 1-5 | linear | 0.383 (n 905) | 0.322 (n 4215; per animal [0.297, 0.287, 0.334, 0.287, 0.339, 0.32, 0.307, 0.386, 0.348, 0.316]) |
| 1-5 | nonlinear | 0.299 (n 905) | 0.286 (n 4215; per animal [0.248, 0.223, 0.31, 0.254, 0.284, 0.298, 0.282, 0.35, 0.326, 0.288]) |

Focus edits on twin streams (edit at chunk 2, 3 or 4; 384 pairs; 95 % intervals):

| edit | A's new colour | B's new colour | A's previous colour | next chunk: A / B |
|---|---|---|---|---|
| no edit | 0.286 [0.24, 0.33] | 0.297 [0.25, 0.34] | 0.083 [0.06, 0.12] | 0.286 / 0.234 |
| focus A → B before the chunk is read | 0.279 [0.24, 0.33] | 0.318 [0.27, 0.37] | 0.086 [0.06, 0.12] | 0.299 / 0.242 |
| same edit after the chunk is read (question only) | 0.255 [0.21, 0.30] | 0.328 [0.28, 0.38] | 0.096 [0.07, 0.13] | 0.286 / 0.234 |
| half strength | 0.276 [0.23, 0.32] | 0.315 [0.27, 0.36] | 0.083 [0.06, 0.12] | 0.286 / 0.232 |
| random direction, same size | 0.286 [0.24, 0.33] | 0.302 [0.26, 0.35] | 0.081 [0.06, 0.11] | 0.292 / 0.242 |
| toward an animal absent from the chunk | 0.271 [0.23, 0.32] | 0.312 [0.27, 0.36] | 0.089 [0.06, 0.12] | 0.292 / 0.232 |

The frozen model's own attention from the chunk's last token while it reads chunk 3 (mean over heads and over layers 13-24; twin A streams):

| condition | on A's record | on B's record | on each other record | on the placeholders |
|---|---|---|---|---|
| trained state | 0.1404 | 0.1301 | 0.1372 | 0.1383 |
| focus edit A → B | 0.1405 | 0.1301 | 0.1372 | 0.1382 |
| no memory (initial state) | 0.1309 | 0.1206 | 0.1288 | 0.1762 |
| nothing written | 0.1534 | 0.1385 | 0.1474 | 0.0202 |

Per layer, A / B (trained state → after the edit): 13: 0.087/0.073 → 0.087/0.073; 14: 0.153/0.134 → 0.154/0.134; 15: 0.157/0.141 → 0.157/0.140; 16: 0.137/0.120 → 0.137/0.120; 17: 0.094/0.092 → 0.094/0.092; 18: 0.168/0.149 → 0.168/0.150; 19: 0.112/0.107 → 0.112/0.108; 20: 0.128/0.124 → 0.128/0.124; 21: 0.147/0.155 → 0.147/0.154; 22: 0.173/0.171 → 0.172/0.171; 23: 0.139/0.125 → 0.140/0.125; 24: 0.190/0.169 → 0.190/0.169

Chunk 0, instruction in view (A's stream says A, its twin says B; both records are in the chunk): attention from the last token

| instruction | write | on A's record | on B's record | on each other record | on the placeholders |
|---|---|---|---|---|---|
| A | learned write | 0.1822 | 0.0958 | 0.1053 | 0.1746 |
| A | nothing written | 0.1548 | 0.1307 | 0.1511 | 0.0183 |
| B | learned write | 0.1070 | 0.1710 | 0.1054 | 0.1738 |
| B | nothing written | 0.1509 | 0.1343 | 0.1497 | 0.0186 |

