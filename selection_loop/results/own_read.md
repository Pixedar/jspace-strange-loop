# Reads that can fail (section 5.4)

| arm | overall | tracked updated | colour kept | reads per chunk | first read on tracked (present) | first read "nothing new" (absent) |
|---|---|---|---|---|---|---|
| blind_c0.1_ws | 0.909 | 0.921 | 0.870 | 1.49 | 0.956 | 0.923 |
| blind_c0.1_ws_s1 | 0.901 | 0.916 | 0.849 | 1.52 | 0.966 | 0.909 |
| retry_c0.1_nonull_ws | 0.881 | 0.911 | 0.781 | 1.33 | 0.957 | 0.946 |
| retry_c0.1_ws | 0.905 | 0.916 | 0.867 | 1.32 | 0.960 | 0.934 |
| retry_c0.1_ws_s1 | 0.901 | 0.915 | 0.856 | 1.33 | 0.961 | 0.937 |
| single_c0.1 | 0.657 | 0.691 | 0.545 | 0.74 | 0.835 | 0.752 |
| single_c0.1_ws | 0.721 | 0.719 | 0.728 | 0.76 | 0.957 | 0.920 |

Second read, by what happened to the first read (attention on the tracked record / on "nothing new"; accuracy at that question):

| arm | first read failed | first read delivered | tracked animal absent |
|---|---|---|---|
| blind_c0.1_ws | 0.937 / 0.060; acc 0.743 (n 724) | 0.932 / 0.064; acc 0.999 (n 1640) | 0.000 / 0.948; acc 0.870 (n 707) |
| blind_c0.1_ws_s1 | 0.953 / 0.042; acc 0.732 (n 723) | 0.951 / 0.045; acc 0.998 (n 1640) | 0.000 / 0.937; acc 0.849 (n 707) |
| retry_c0.1_nonull_ws | 0.963 / 0.036; acc 0.729 (n 713) | 0.643 / 0.356; acc 0.996 (n 1622) | 0.000 / 0.989; acc 0.781 (n 707) |
| retry_c0.1_ws | 0.951 / 0.048; acc 0.736 (n 720) | 0.621 / 0.378; acc 0.998 (n 1623) | 0.000 / 0.987; acc 0.867 (n 707) |
| retry_c0.1_ws_s1 | 0.955 / 0.045; acc 0.732 (n 713) | 0.639 / 0.359; acc 0.997 (n 1627) | 0.000 / 0.987; acc 0.856 (n 707) |
| single_c0.1 | 0.000 / 0.000; acc 0.068 (n 695) | 0.000 / 0.000; acc 0.994 (n 1564) | 0.000 / 0.000; acc 0.545 (n 707) |
| single_c0.1_ws | 0.000 / 0.000; acc 0.081 (n 724) | 0.000 / 0.000; acc 1.000 (n 1640) | 0.000 / 0.000; acc 0.728 (n 707) |

Knowing that it does not know (same tracked animal as before; the stored colour is then the old one, always wrong if the update was missed):

| arm | update missed: acc / max prob / prob on previous true colour | delivered by first read | delivered by second read |
|---|---|---|---|
| blind_c0.1_ws | 0.043 / 0.333 / 0.308 (n 140) | 0.999 / 0.980 / 0.006 (n 1200) | 1.000 / 0.965 / 0.009 (n 358) |
| blind_c0.1_ws_s1 | 0.007 / 0.486 / 0.435 (n 139) | 0.998 / 0.983 / 0.004 (n 1201) | 1.000 / 0.984 / 0.004 (n 359) |
| retry_c0.1_nonull_ws | 0.014 / 0.632 / 0.600 (n 139) | 0.995 / 0.983 / 0.008 (n 1192) | 0.986 / 0.960 / 0.026 (n 356) |
| retry_c0.1_ws | 0.021 / 0.305 / 0.266 (n 141) | 0.998 / 0.987 / 0.004 (n 1200) | 0.997 / 0.969 / 0.011 (n 357) |
| retry_c0.1_ws_s1 | 0.036 / 0.443 / 0.373 (n 138) | 0.997 / 0.986 / 0.004 (n 1197) | 0.994 / 0.975 / 0.009 (n 356) |
| single_c0.1 | 0.057 / 0.239 / 0.204 (n 474) | 0.993 / 0.928 / 0.018 (n 1138) | – |
| single_c0.1_ws | 0.060 / 0.173 / 0.144 (n 498) | 1.000 / 0.992 / 0.002 (n 1201) | – |

| arm | intermediate state: "first read failed" (balanced acc., grouped by stream) | "tried and failed" vs "chose nothing" |
|---|---|---|
| blind_c0.1_ws | 0.994 | 0.950 |
| blind_c0.1_ws_s1 | 0.989 | 0.938 |
| retry_c0.1_nonull_ws | 0.995 | 0.615 |
| retry_c0.1_ws | 0.995 | 0.952 |
| retry_c0.1_ws_s1 | 0.977 | 0.947 |
| single_c0.1 | 0.975 | 0.978 |
| single_c0.1_ws | 0.999 | 0.996 |

Paired counterfactual (same stream, history, first attention and noise; only the first delivery at chunk t differs; pooled over t = 2-4):

| arm | delivered | failed | delivered, second query from the failed state | failed, second query from the delivered state |
|---|---|---|---|---|
| blind_c0.1_ws | second read on tracked 0.910; acc 0.999 (n 1118) | second read on tracked 0.910; acc 0.996 (n 1118) | second read on tracked 0.910; acc 0.999 (n 1118) | second read on tracked 0.910; acc 0.996 (n 1118) |
| blind_c0.1_ws_s1 | second read on tracked 0.933; acc 0.999 (n 1116) | second read on tracked 0.933; acc 0.993 (n 1116) | second read on tracked 0.933; acc 0.999 (n 1116) | second read on tracked 0.933; acc 0.993 (n 1116) |
| retry_c0.1_nonull_ws | second read on tracked 0.581; acc 0.998 (n 1090) | second read on tracked 0.954; acc 0.987 (n 1090) | second read on tracked 0.954; acc 0.999 (n 1090) | second read on tracked 0.581; acc 0.592 (n 1090) |
| retry_c0.1_ws | second read on tracked 0.561; acc 1.000 (n 1097) | second read on tracked 0.936; acc 0.994 (n 1097) | second read on tracked 0.936; acc 0.998 (n 1097) | second read on tracked 0.561; acc 0.624 (n 1097) |
| retry_c0.1_ws_s1 | second read on tracked 0.559; acc 0.999 (n 1097) | second read on tracked 0.940; acc 0.996 (n 1097) | second read on tracked 0.940; acc 0.999 (n 1097) | second read on tracked 0.559; acc 0.530 (n 1097) |

Other failure rates at test time (trained at 0.3):

| arm | rate | acc | reads per chunk | second read on tracked after failed / after delivered |
|---|---|---|---|---|
| blind_c0.1_ws | 0 | 0.994 | 1.50 | – / 0.938 |
| blind_c0.1_ws | 0.6 | 0.657 | 1.48 | 0.930 / 0.928 |
| blind_c0.1_ws_s1 | 0 | 0.991 | 1.51 | – / 0.941 |
| blind_c0.1_ws_s1 | 0.6 | 0.654 | 1.52 | 0.962 / 0.962 |
| retry_c0.1_nonull_ws | 0 | 0.974 | 1.24 | – / 0.627 |
| retry_c0.1_nonull_ws | 0.6 | 0.637 | 1.42 | 0.973 / 0.671 |
| retry_c0.1_ws | 0 | 0.993 | 1.23 | – / 0.603 |
| retry_c0.1_ws | 0.6 | 0.653 | 1.41 | 0.958 / 0.642 |
| retry_c0.1_ws_s1 | 0 | 0.989 | 1.24 | – / 0.613 |
| retry_c0.1_ws_s1 | 0.6 | 0.652 | 1.42 | 0.961 / 0.665 |
| single_c0.1 | 0 | 0.900 | 0.73 | – / 0.000 |
| single_c0.1 | 0.6 | 0.419 | 0.74 | 0.000 / 0.000 |
| single_c0.1_ws | 0 | 0.993 | 0.75 | – / 0.000 |
| single_c0.1_ws | 0.6 | 0.449 | 0.76 | 0.000 / 0.000 |

### Edits of the intermediate state: blind_c0.1_ws (direction norm 0.371; calibration {'failed': 733, 'delivered': 1643}; one chunk at a time, chunks 2-4, mean of the per-chunk means)

| edit | moved (1-cos) | first read failed: second read on tracked / acc | first read delivered: second read on tracked / acc | absent: second read on tracked / "nothing new" |
|---|---|---|---|---|
| no edit | 0.000 | 0.912 / 0.744 | 0.909 / 0.999 | 0.000 / 0.945 |
| toward "failed" | 0.053 | 0.912 / 0.741 | 0.909 / 0.991 | 0.000 / 0.945 |
| toward "failed" × 0.5 | 0.015 | 0.912 / 0.741 | 0.909 / 0.999 | 0.000 / 0.945 |
| toward "failed" × 2 | 0.160 | 0.912 / 0.744 | 0.909 / 0.938 | 0.000 / 0.945 |
| toward "failed", second query only | 0.053 | 0.912 / 0.744 | 0.909 / 0.999 | 0.000 / 0.945 |
| toward "failed", second update only | 0.053 | 0.912 / 0.741 | 0.909 / 0.991 | 0.000 / 0.945 |
| random, same size (mean of 3) | 0.062 | 0.912 / 0.742 | 0.909 / 0.994 | 0.000 / 0.945 |
| toward "delivered" | 0.067 | 0.912 / 0.710 | 0.909 / 0.999 | 0.000 / 0.945 |
| toward "delivered", second query only | 0.067 | 0.912 / 0.744 | 0.909 / 0.999 | 0.000 / 0.945 |
| toward "delivered", second update only | 0.067 | 0.912 / 0.710 | 0.909 / 0.999 | 0.000 / 0.945 |

### Edits of the intermediate state: blind_c0.1_ws_s1 (direction norm 0.232; calibration {'failed': 735, 'delivered': 1640}; one chunk at a time, chunks 2-4, mean of the per-chunk means)

| edit | moved (1-cos) | first read failed: second read on tracked / acc | first read delivered: second read on tracked / acc | absent: second read on tracked / "nothing new" |
|---|---|---|---|---|
| no edit | 0.000 | 0.931 / 0.732 | 0.935 / 0.997 | 0.000 / 0.934 |
| toward "failed" | 0.020 | 0.931 / 0.732 | 0.935 / 0.996 | 0.000 / 0.934 |
| toward "failed" × 0.5 | 0.005 | 0.931 / 0.732 | 0.935 / 0.996 | 0.000 / 0.934 |
| toward "failed" × 2 | 0.065 | 0.931 / 0.735 | 0.935 / 0.994 | 0.000 / 0.934 |
| toward "failed", second query only | 0.020 | 0.931 / 0.732 | 0.935 / 0.997 | 0.000 / 0.934 |
| toward "failed", second update only | 0.020 | 0.931 / 0.732 | 0.935 / 0.996 | 0.000 / 0.934 |
| random, same size (mean of 3) | 0.026 | 0.931 / 0.734 | 0.935 / 0.996 | 0.000 / 0.934 |
| toward "delivered" | 0.026 | 0.931 / 0.735 | 0.935 / 0.993 | 0.000 / 0.934 |
| toward "delivered", second query only | 0.026 | 0.931 / 0.732 | 0.935 / 0.997 | 0.000 / 0.934 |
| toward "delivered", second update only | 0.026 | 0.931 / 0.735 | 0.935 / 0.993 | 0.000 / 0.934 |

### Edits of the intermediate state: retry_c0.1_nonull_ws (direction norm 0.215; calibration {'failed': 727, 'delivered': 1621}; one chunk at a time, chunks 2-4, mean of the per-chunk means)

| edit | moved (1-cos) | first read failed: second read on tracked / acc | first read delivered: second read on tracked / acc | absent: second read on tracked / "nothing new" |
|---|---|---|---|---|
| no edit | 0.000 | 0.952 / 0.719 | 0.577 / 0.992 | 0.000 / 0.989 |
| toward "failed" | 0.022 | 0.952 / 0.728 | 0.630 / 0.995 | 0.000 / 0.983 |
| toward "failed" × 0.5 | 0.006 | 0.953 / 0.722 | 0.608 / 0.993 | 0.000 / 0.987 |
| toward "failed" × 2 | 0.083 | 0.936 / 0.728 | 0.643 / 0.983 | 0.000 / 0.972 |
| toward "failed", second query only | 0.022 | 0.952 / 0.719 | 0.630 / 0.992 | 0.000 / 0.983 |
| toward "failed", second update only | 0.022 | 0.952 / 0.728 | 0.577 / 0.993 | 0.000 / 0.989 |
| random, same size (mean of 3) | 0.022 | 0.948 / 0.721 | 0.592 / 0.990 | 0.000 / 0.984 |
| toward "delivered" | 0.022 | 0.933 / 0.710 | 0.496 / 0.989 | 0.000 / 0.992 |
| toward "delivered", second query only | 0.022 | 0.933 / 0.710 | 0.496 / 0.991 | 0.000 / 0.992 |
| toward "delivered", second update only | 0.022 | 0.952 / 0.716 | 0.577 / 0.991 | 0.000 / 0.989 |

### Edits of the intermediate state: retry_c0.1_ws (direction norm 0.299; calibration {'failed': 727, 'delivered': 1626}; one chunk at a time, chunks 2-4, mean of the per-chunk means)

| edit | moved (1-cos) | first read failed: second read on tracked / acc | first read delivered: second read on tracked / acc | absent: second read on tracked / "nothing new" |
|---|---|---|---|---|
| no edit | 0.000 | 0.936 / 0.734 | 0.555 / 0.997 | 0.000 / 0.988 |
| toward "failed" | 0.040 | 0.912 / 0.737 | 0.606 / 0.990 | 0.000 / 0.981 |
| toward "failed" × 0.5 | 0.011 | 0.928 / 0.740 | 0.589 / 0.997 | 0.000 / 0.985 |
| toward "failed" × 2 | 0.133 | 0.857 / 0.737 | 0.595 / 0.960 | 0.000 / 0.972 |
| toward "failed", second query only | 0.040 | 0.912 / 0.737 | 0.606 / 0.997 | 0.000 / 0.981 |
| toward "failed", second update only | 0.040 | 0.936 / 0.737 | 0.555 / 0.990 | 0.000 / 0.988 |
| random, same size (mean of 3) | 0.041 | 0.919 / 0.739 | 0.567 / 0.996 | 0.000 / 0.979 |
| toward "delivered" | 0.043 | 0.926 / 0.736 | 0.446 / 0.997 | 0.000 / 0.992 |
| toward "delivered", second query only | 0.043 | 0.926 / 0.731 | 0.446 / 0.997 | 0.000 / 0.992 |
| toward "delivered", second update only | 0.043 | 0.936 / 0.740 | 0.555 / 0.999 | 0.000 / 0.988 |

### Edits of the intermediate state: retry_c0.1_ws_s1 (direction norm 0.164; calibration {'failed': 731, 'delivered': 1627}; one chunk at a time, chunks 2-4, mean of the per-chunk means)

| edit | moved (1-cos) | first read failed: second read on tracked / acc | first read delivered: second read on tracked / acc | absent: second read on tracked / "nothing new" |
|---|---|---|---|---|
| no edit | 0.000 | 0.940 / 0.730 | 0.567 / 0.994 | 0.000 / 0.987 |
| toward "failed" | 0.012 | 0.946 / 0.730 | 0.626 / 0.992 | 0.000 / 0.979 |
| toward "failed" × 0.5 | 0.003 | 0.944 / 0.730 | 0.599 / 0.994 | 0.000 / 0.984 |
| toward "failed" × 2 | 0.044 | 0.942 / 0.733 | 0.665 / 0.995 | 0.000 / 0.965 |
| toward "failed", second query only | 0.012 | 0.946 / 0.730 | 0.626 / 0.994 | 0.000 / 0.979 |
| toward "failed", second update only | 0.012 | 0.940 / 0.727 | 0.567 / 0.994 | 0.000 / 0.987 |
| random, same size (mean of 3) | 0.013 | 0.929 / 0.728 | 0.537 / 0.994 | 0.000 / 0.988 |
| toward "delivered" | 0.013 | 0.923 / 0.728 | 0.493 / 0.994 | 0.000 / 0.992 |
| toward "delivered", second query only | 0.013 | 0.923 / 0.728 | 0.493 / 0.994 | 0.000 / 0.992 |
| toward "delivered", second update only | 0.013 | 0.940 / 0.730 | 0.567 / 0.994 | 0.000 / 0.987 |
