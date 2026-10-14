# Edit controls (edits at chunks 2, 3 and 4; 256 twin streams each)

### sel-state_l0 (edits_sel-state_l0.json)

Averaged over the edit at chunks 2, 3, 4 (256 twin streams each).

| condition | A's new colour | B's new colour | C's colour | A's stored colour | attention A | attention B | attention C | attention nothing new | next chunk: A | next chunk: B | state moved (1-cos) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.999 | 0.000 | 0.095 | 0.001 | 0.765 | 0.012 | 0.012 | 0.200 | 1.000 | 0.089 | – |
| focus code A→B | 0.000 | 0.996 | 0.104 | 0.099 | 0.010 | 0.777 | 0.011 | 0.192 | 0.089 | 0.996 | 0.279 |
| same edit, query only | 0.000 | 0.996 | 0.104 | 0.100 | 0.010 | 0.777 | 0.011 | 0.192 | 0.694 | 0.266 | – |
| same edit, answer prompt only | 0.997 | 0.001 | 0.095 | 0.003 | 0.765 | 0.012 | 0.012 | 0.200 | 0.193 | 0.326 | – |
| focus edit × 0.5 | 0.099 | 0.238 | 0.109 | 0.738 | 0.180 | 0.209 | 0.021 | 0.571 | 0.344 | 0.297 | 0.073 |
| focus edit × 2 | 0.000 | 0.996 | 0.103 | 0.096 | 0.000 | 0.961 | 0.010 | 0.018 | 0.083 | 0.949 | 0.677 |
| focus edit reversed | 0.996 | 0.000 | 0.098 | 0.001 | 0.960 | 0.000 | 0.013 | 0.018 | 0.896 | 0.085 | 0.117 |
| random edit, same size | 0.816 | 0.025 | 0.115 | 0.124 | 0.611 | 0.027 | 0.033 | 0.297 | 0.688 | 0.087 | 0.194 |
| focus code A→C (C in the chunk) | 0.095 | 0.104 | 0.995 | 0.107 | 0.009 | 0.011 | 0.778 | 0.191 | 0.111 | 0.115 | 0.284 |
| focus code A→D (D not in the chunk) | 0.003 | 0.102 | 0.100 | 0.984 | 0.042 | 0.048 | 0.046 | 0.814 | 0.099 | 0.108 | 0.279 |

Colour edit (tracked animal not updated at the edited chunk): own_answers_old 0.988, edited_answers_old 0.000, edited_answers_new 0.996, next_chunk_acquires_tracked_update 1.000, own_next_chunk_acquires 0.996, random_same_size_answers_old 0.738, random_same_size_answers_new 0.023

### sel-state_l0_s1 (edits_sel-state_l0_s1.json)

Averaged over the edit at chunks 2, 3, 4 (256 twin streams each).

| condition | A's new colour | B's new colour | C's colour | A's stored colour | attention A | attention B | attention C | attention nothing new | next chunk: A | next chunk: B | state moved (1-cos) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.997 | 0.000 | 0.095 | 0.003 | 0.835 | 0.009 | 0.010 | 0.137 | 0.993 | 0.090 | – |
| focus code A→B | 0.000 | 0.997 | 0.103 | 0.099 | 0.006 | 0.854 | 0.008 | 0.123 | 0.086 | 0.995 | 0.305 |
| same edit, query only | 0.000 | 0.996 | 0.104 | 0.100 | 0.006 | 0.854 | 0.008 | 0.123 | 0.702 | 0.255 | – |
| same edit, answer prompt only | 0.996 | 0.000 | 0.095 | 0.004 | 0.835 | 0.009 | 0.010 | 0.137 | 0.190 | 0.329 | – |
| focus edit × 0.5 | 0.171 | 0.285 | 0.091 | 0.609 | 0.211 | 0.246 | 0.020 | 0.502 | 0.372 | 0.315 | 0.080 |
| focus edit × 2 | 0.000 | 1.000 | 0.103 | 0.096 | 0.000 | 0.980 | 0.006 | 0.008 | 0.085 | 0.977 | 0.721 |
| focus edit reversed | 1.000 | 0.000 | 0.095 | 0.000 | 0.979 | 0.000 | 0.007 | 0.009 | 0.973 | 0.087 | 0.121 |
| random edit, same size | 0.835 | 0.021 | 0.111 | 0.096 | 0.665 | 0.029 | 0.030 | 0.242 | 0.703 | 0.095 | 0.207 |
| focus code A→C (C in the chunk) | 0.094 | 0.103 | 0.995 | 0.107 | 0.006 | 0.008 | 0.856 | 0.121 | 0.113 | 0.116 | 0.305 |
| focus code A→D (D not in the chunk) | 0.007 | 0.098 | 0.108 | 0.952 | 0.042 | 0.048 | 0.052 | 0.803 | 0.109 | 0.105 | 0.306 |

Colour edit (tracked animal not updated at the edited chunk): own_answers_old 0.969, edited_answers_old 0.004, edited_answers_new 0.992, next_chunk_acquires_tracked_update 0.996, own_next_chunk_acquires 1.000, random_same_size_answers_old 0.707, random_same_size_answers_new 0.027

### sel-state_l0_s2 (edits_sel-state_l0_s2.json)

Averaged over the edit at chunks 2, 3, 4 (256 twin streams each).

| condition | A's new colour | B's new colour | C's colour | A's stored colour | attention A | attention B | attention C | attention nothing new | next chunk: A | next chunk: B | state moved (1-cos) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.997 | 0.000 | 0.095 | 0.003 | 0.829 | 0.008 | 0.009 | 0.146 | 0.990 | 0.089 | – |
| focus code A→B | 0.000 | 0.996 | 0.104 | 0.100 | 0.006 | 0.837 | 0.008 | 0.141 | 0.089 | 0.988 | 0.278 |
| same edit, query only | 0.000 | 0.995 | 0.103 | 0.102 | 0.006 | 0.837 | 0.008 | 0.141 | 0.699 | 0.251 | – |
| same edit, answer prompt only | 0.996 | 0.000 | 0.095 | 0.004 | 0.829 | 0.008 | 0.009 | 0.146 | 0.190 | 0.328 | – |
| focus edit × 0.5 | 0.148 | 0.260 | 0.108 | 0.632 | 0.202 | 0.234 | 0.019 | 0.526 | 0.322 | 0.290 | 0.072 |
| focus edit × 2 | 0.000 | 1.000 | 0.103 | 0.096 | 0.000 | 0.983 | 0.004 | 0.008 | 0.090 | 0.954 | 0.678 |
| focus edit reversed | 1.000 | 0.000 | 0.095 | 0.000 | 0.984 | 0.000 | 0.004 | 0.008 | 0.949 | 0.082 | 0.116 |
| random edit, same size | 0.826 | 0.021 | 0.105 | 0.079 | 0.663 | 0.027 | 0.029 | 0.249 | 0.647 | 0.081 | 0.193 |
| focus code A→C (C in the chunk) | 0.095 | 0.103 | 0.991 | 0.109 | 0.007 | 0.007 | 0.835 | 0.143 | 0.112 | 0.111 | 0.278 |
| focus code A→D (D not in the chunk) | 0.017 | 0.107 | 0.104 | 0.923 | 0.041 | 0.042 | 0.045 | 0.827 | 0.109 | 0.100 | 0.278 |

Colour edit (tracked animal not updated at the edited chunk): own_answers_old 0.898, edited_answers_old 0.000, edited_answers_new 0.934, next_chunk_acquires_tracked_update 0.996, own_next_chunk_acquires 1.000, random_same_size_answers_old 0.707, random_same_size_answers_new 0.023

### sel-state_l0_s1 (edits_sel-state_l0_s1.json)

Averaged over the edit at chunks 2, 3, 4 (256 twin streams each).

| condition | A's new colour | B's new colour | C's colour | A's stored colour | attention A | attention B | attention C | attention nothing new | next chunk: A | next chunk: B | state moved (1-cos) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.995 | 0.004 | 0.098 | 0.001 | 0.995 | 0.003 | 0.000 | 0.000 | 0.793 | 0.133 | – |
| focus code A→B | 0.005 | 0.987 | 0.103 | 0.100 | 0.004 | 0.991 | 0.001 | 0.000 | 0.181 | 0.495 | 0.661 |
| same edit, query only | 0.005 | 0.992 | 0.103 | 0.095 | 0.004 | 0.991 | 0.001 | 0.000 | 0.751 | 0.163 | – |
| same edit, answer prompt only | 0.990 | 0.004 | 0.096 | 0.005 | 0.995 | 0.003 | 0.000 | 0.000 | 0.198 | 0.378 | – |
| focus edit × 0.5 | 0.499 | 0.372 | 0.130 | 0.092 | 0.497 | 0.388 | 0.048 | 0.001 | 0.479 | 0.238 | 0.179 |
| focus edit × 2 | 0.000 | 0.969 | 0.121 | 0.098 | 0.000 | 0.965 | 0.020 | 0.000 | 0.126 | 0.460 | 1.126 |
| focus edit reversed | 0.969 | 0.004 | 0.115 | 0.004 | 0.967 | 0.000 | 0.020 | 0.000 | 0.758 | 0.118 | 0.115 |
| random edit, same size | 0.928 | 0.020 | 0.126 | 0.014 | 0.927 | 0.012 | 0.023 | 0.000 | 0.656 | 0.129 | 0.325 |
| focus code A→C (C in the chunk) | 0.095 | 0.102 | 0.987 | 0.109 | 0.001 | 0.001 | 0.997 | 0.000 | 0.198 | 0.152 | 0.672 |
| focus code A→D (D not in the chunk) | 0.271 | 0.279 | 0.290 | 0.154 | 0.231 | 0.222 | 0.207 | 0.125 | 0.191 | 0.182 | 0.666 |

Colour edit (tracked animal not updated at the edited chunk): own_answers_old 0.223, edited_answers_old 0.113, edited_answers_new 0.199, next_chunk_acquires_tracked_update 0.984, own_next_chunk_acquires 1.000, random_same_size_answers_old 0.215, random_same_size_answers_new 0.090

### sel-const_l0 (edits_sel-const_l0.json)

Averaged over the edit at chunks 2, 3, 4 (256 twin streams each).

| condition | A's new colour | B's new colour | C's colour | A's stored colour | attention A | attention B | attention C | attention nothing new | next chunk: A | next chunk: B | state moved (1-cos) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.320 | 0.324 | 0.395 | 0.124 | 0.245 | 0.245 | 0.247 | 0.018 | 0.307 | 0.238 | – |
| focus code A→B | 0.324 | 0.328 | 0.389 | 0.121 | 0.245 | 0.245 | 0.247 | 0.018 | 0.315 | 0.237 | 0.001 |
| focus edit × 0.5 | 0.323 | 0.327 | 0.385 | 0.122 | 0.245 | 0.245 | 0.247 | 0.018 | 0.320 | 0.240 | 0.000 |
| focus edit × 2 | 0.331 | 0.328 | 0.382 | 0.122 | 0.245 | 0.245 | 0.247 | 0.018 | 0.312 | 0.242 | 0.004 |
| focus edit reversed | 0.326 | 0.322 | 0.387 | 0.122 | 0.245 | 0.245 | 0.247 | 0.018 | 0.314 | 0.242 | 0.001 |
| random edit, same size | 0.318 | 0.333 | 0.384 | 0.120 | 0.245 | 0.245 | 0.247 | 0.018 | 0.309 | 0.238 | 0.001 |
| focus code A→C (C in the chunk) | 0.319 | 0.327 | 0.389 | 0.125 | 0.245 | 0.245 | 0.247 | 0.018 | 0.311 | 0.230 | 0.001 |
| focus code A→D (D not in the chunk) | 0.322 | 0.332 | 0.385 | 0.122 | 0.245 | 0.245 | 0.247 | 0.018 | 0.315 | 0.243 | 0.001 |

Colour edit (tracked animal not updated at the edited chunk): own_answers_old 0.164, edited_answers_old 0.121, edited_answers_new 0.172, next_chunk_acquires_tracked_update 0.453, own_next_chunk_acquires 0.445, random_same_size_answers_old 0.176, random_same_size_answers_new 0.102

### sel-fcst_l0.05 (edits_sel-fcst_l0.05.json)

Averaged over the edit at chunks 2, 3, 4 (256 twin streams each).

| condition | A's new colour | B's new colour | C's colour | A's stored colour | attention A | attention B | attention C | attention nothing new | next chunk: A | next chunk: B | state moved (1-cos) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.805 | 0.081 | 0.173 | 0.039 | 0.465 | 0.110 | 0.119 | 0.204 | 0.642 | 0.148 | – |
| focus code A→B | 0.095 | 0.788 | 0.171 | 0.121 | 0.116 | 0.458 | 0.107 | 0.210 | 0.249 | 0.375 | 0.371 |
| same edit, query only | 0.094 | 0.811 | 0.163 | 0.107 | 0.116 | 0.458 | 0.107 | 0.210 | 0.621 | 0.188 | – |
| same edit, answer prompt only | 0.799 | 0.085 | 0.174 | 0.033 | 0.465 | 0.110 | 0.119 | 0.204 | 0.258 | 0.297 | – |
| focus edit × 0.5 | 0.341 | 0.331 | 0.306 | 0.104 | 0.233 | 0.233 | 0.143 | 0.255 | 0.368 | 0.198 | 0.101 |
| focus edit × 2 | 0.077 | 0.673 | 0.232 | 0.113 | 0.077 | 0.448 | 0.123 | 0.214 | 0.210 | 0.327 | 0.698 |
| focus edit reversed | 0.702 | 0.068 | 0.242 | 0.048 | 0.447 | 0.072 | 0.141 | 0.217 | 0.467 | 0.129 | 0.087 |
| random edit, same size | 0.686 | 0.139 | 0.217 | 0.047 | 0.392 | 0.139 | 0.128 | 0.219 | 0.512 | 0.151 | 0.198 |
| focus code A→C (C in the chunk) | 0.161 | 0.158 | 0.803 | 0.126 | 0.124 | 0.105 | 0.453 | 0.209 | 0.245 | 0.171 | 0.353 |
| focus code A→D (D not in the chunk) | 0.263 | 0.281 | 0.327 | 0.155 | 0.164 | 0.175 | 0.172 | 0.325 | 0.201 | 0.180 | 0.396 |

Colour edit (tracked animal not updated at the edited chunk): own_answers_old 0.223, edited_answers_old 0.102, edited_answers_new 0.246, next_chunk_acquires_tracked_update 0.809, own_next_chunk_acquires 0.824, random_same_size_answers_old 0.207, random_same_size_answers_new 0.113
