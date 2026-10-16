### Exp A (ambient noise), mean max principal angle in degrees, default hyper-parameters: by noise level (avg over manifolds)

|   sigma |   lpca |   ldle |   lego |
|--------:|-------:|-------:|-------:|
|   0     |   1.57 |   3.63 |   9.93 |
|   0.005 |   6.75 |  14.84 |  10.29 |
|   0.01  |  12.04 |  23.5  |   9.85 |
|   0.02  |  20.76 |  34.98 |  11.07 |
|   0.03  |  28.12 |  43.23 |  12.95 |
|   0.05  |  39.47 |  54.12 |  16.71 |

### Exp A, default hyper-parameters: by manifold (avg over noise levels)

| manifold    |   lpca |   ldle |   lego |
|:------------|-------:|-------:|-------:|
| circle      |  26.25 |  25.8  |  10.88 |
| clifford T2 |  11.63 |  22.05 |   8.3  |
| s-curve     |  22.18 |  34.35 |  13.99 |
| saddle      |  21.87 |  34.79 |  10.71 |
| sphere S2   |  13.28 |  25.15 |   8.05 |
| sphere S3   |  10.23 |  36.94 |  18.82 |
| swiss roll  |  18.86 |  30.99 |  13.04 |
| torus R3    |  16.4  |  28.62 |  12.08 |
| trefoil     |  22.35 |  22.75 |  10.33 |

### Exp A, default hyper-parameters: sigma=0.0

| manifold    |   lpca |   ldle |   lego |
|:------------|-------:|-------:|-------:|
| circle      |   0.12 |   0.11 |   4.5  |
| clifford T2 |   2.61 |   3.49 |   7.21 |
| s-curve     |   1.65 |   2.35 |  12.21 |
| saddle      |   1.1  |   1.2  |   4.12 |
| sphere S2   |   1.12 |   2.29 |   7.03 |
| sphere S3   |   2.59 |  16.98 |  23.12 |
| swiss roll  |   1.85 |   2.48 |  11.72 |
| torus R3    |   2.83 |   3.51 |  12.13 |
| trefoil     |   0.27 |   0.25 |   7.3  |

### Exp A, default hyper-parameters: sigma=0.02

| manifold    |   lpca |   ldle |   lego |
|:------------|-------:|-------:|-------:|
| circle      |  34.16 |  32.53 |  13.02 |
| clifford T2 |  11.66 |  24.78 |   7.83 |
| s-curve     |  25.18 |  42.93 |  12.41 |
| saddle      |  25.38 |  43.93 |  10.81 |
| sphere S2   |  14.02 |  28.76 |   7.18 |
| sphere S3   |  10    |  41.81 |  16.33 |
| swiss roll  |  20.22 |  37.43 |  11.08 |
| torus R3    |  17.19 |  32.78 |  10.57 |
| trefoil     |  29.02 |  29.83 |  10.39 |

### Exp A, default hyper-parameters: sigma=0.05

| manifold    |   lpca |   ldle |   lego |
|:------------|-------:|-------:|-------:|
| circle      |  43.48 |  39.82 |  16.19 |
| clifford T2 |  28.12 |  47.28 |  11.4  |
| s-curve     |  50.83 |  63.02 |  21.99 |
| saddle      |  49.82 |  64.45 |  18.59 |
| sphere S2   |  32.91 |  53.73 |  11.94 |
| sphere S3   |  25.03 |  60.87 |  16.41 |
| swiss roll  |  45.71 |  61.36 |  21.45 |
| torus R3    |  38.99 |  60.06 |  16.97 |
| trefoil     |  40.32 |  36.5  |  15.43 |

### Exp B (tubular noise), default hyper-parameters

|                        |   lpca |   ldle |   lego |
|:-----------------------|-------:|-------:|-------:|
| ('clifford T2', 0.0)   |   2.61 |   3.49 |   7.21 |
| ('clifford T2', 0.005) |   3.83 |   8.44 |   7.2  |
| ('clifford T2', 0.01)  |   6.15 |  14.74 |   7.2  |
| ('clifford T2', 0.02)  |  11.49 |  24.59 |   7.44 |
| ('clifford T2', 0.03)  |  17.04 |  32.5  |   8.2  |
| ('clifford T2', 0.05)  |  27.48 |  46.72 |  10.25 |
| ('sphere S2', 0.0)     |   1.12 |   2.29 |   7.03 |
| ('sphere S2', 0.005)   |   3.64 |  10.01 |   6.94 |
| ('sphere S2', 0.01)    |   7.03 |  17.53 |   6.73 |
| ('sphere S2', 0.02)    |  14.03 |  29.04 |   7.03 |
| ('sphere S2', 0.03)    |  21.01 |  38.28 |   8.27 |
| ('sphere S2', 0.05)    |  33.06 |  53.78 |  11.38 |
| ('swiss roll', 0.0)    |   1.85 |   2.48 |  11.72 |
| ('swiss roll', 0.005)  |   5.42 |  13.9  |  10.39 |
| ('swiss roll', 0.01)   |  10.28 |  23.7  |   9.71 |
| ('swiss roll', 0.02)   |  20.49 |  37.55 |  10.91 |
| ('swiss roll', 0.03)   |  30.39 |  47.61 |  13.62 |
| ('swiss roll', 0.05)   |  46.46 |  61.47 |  20.46 |

### Exp C (sample size, sigma=0.02), default hyper-parameters

|                       |   lpca |   ldle |   lego |
|:----------------------|-------:|-------:|-------:|
| ('clifford T2', 500)  |   8.33 |  43.06 |  22.44 |
| ('clifford T2', 1000) |   9.03 |  35.8  |  11.44 |
| ('clifford T2', 2000) |  11.66 |  24.78 |   7.83 |
| ('clifford T2', 4000) |  15.88 |  25.9  |   5.59 |
| ('sphere S2', 500)    |   7.78 |  46.4  |  19.81 |
| ('sphere S2', 1000)   |  10.16 |  40.08 |  10.73 |
| ('sphere S2', 2000)   |  14.02 |  28.76 |   7.18 |
| ('sphere S2', 4000)   |  19.58 |  29.33 |   4.99 |
| ('swiss roll', 500)   |  16.93 |  44.92 |  28.36 |
| ('swiss roll', 1000)  |  14.92 |  41.73 |  17.53 |
| ('swiss roll', 2000)  |  20.22 |  37.43 |  11.08 |
| ('swiss roll', 4000)  |  28.13 |  37.42 |   8.27 |

### Exp D (ambient dimension, fixed noise energy), default hyper-parameters

|                     |   lpca |   ldle |   lego |
|:--------------------|-------:|-------:|-------:|
| ('sphere S2', 5)    |  13.91 |  24.33 |   6.84 |
| ('sphere S2', 10)   |  14.01 |  27.72 |   7.05 |
| ('sphere S2', 20)   |  14.1  |  28.99 |   7.19 |
| ('sphere S2', 50)   |  14.05 |  29.96 |   7.14 |
| ('sphere S2', 100)  |  14    |  30.27 |   7.04 |
| ('swiss roll', 5)   |  21.85 |  29.58 |  11.19 |
| ('swiss roll', 10)  |  21.06 |  35.19 |  10.78 |
| ('swiss roll', 20)  |  20.34 |  37.56 |  11.1  |
| ('swiss roll', 50)  |  20.1  |  38.54 |  11.35 |
| ('swiss roll', 100) |  19.94 |  39.3  |  11.1  |

### Exp A (ambient noise), mean max principal angle in degrees, tuned per (manifold, noise) on ground truth, averaged over seeds: by noise level (avg over manifolds)

|   sigma |   lpca |   ldle |   lego |
|--------:|-------:|-------:|-------:|
|   0     |   1.42 |   2.95 |   8.19 |
|   0.005 |   2.44 |  13.12 |   8.34 |
|   0.01  |   3.47 |  20.68 |   7.79 |
|   0.02  |   5.66 |  31.94 |   8.21 |
|   0.03  |   7.99 |  39.84 |   9.47 |
|   0.05  |  12.84 |  50.71 |  12.9  |

### Exp A, tuned per (manifold, noise) on ground truth, averaged over seeds: by manifold (avg over noise levels)

| manifold    |   lpca |   ldle |   lego |
|:------------|-------:|-------:|-------:|
| circle      |   5.67 |  19.09 |   5.71 |
| clifford T2 |   4.24 |  21.43 |   7.89 |
| s-curve     |   7.08 |  31.82 |  10.34 |
| saddle      |   5.95 |  32.86 |   7.12 |
| sphere S2   |   3.47 |  24.5  |   6.18 |
| sphere S3   |   3.69 |  36.35 |  16.99 |
| swiss roll  |   9.79 |  28.86 |  11.32 |
| torus R3    |   6.25 |  27.68 |  11.65 |
| trefoil     |   4.6  |  16.31 |   5.15 |

### Exp A, tuned per (manifold, noise) on ground truth, averaged over seeds: sigma=0.0

| manifold    |   lpca |   ldle |   lego |
|:------------|-------:|-------:|-------:|
| circle      |   0.1  |   0.07 |   1.59 |
| clifford T2 |   2.47 |   2.7  |   7.21 |
| s-curve     |   1.59 |   1.85 |   9.85 |
| saddle      |   1.09 |   0.93 |   4.09 |
| sphere S2   |   1.01 |   1.73 |   5.92 |
| sphere S3   |   1.78 |  14.25 |  19.75 |
| swiss roll  |   1.77 |   1.98 |  10.98 |
| torus R3    |   2.77 |   2.91 |  11.63 |
| trefoil     |   0.22 |   0.16 |   2.73 |

### Exp A, tuned per (manifold, noise) on ground truth, averaged over seeds: sigma=0.02

| manifold    |   lpca |   ldle |   lego |
|:------------|-------:|-------:|-------:|
| circle      |   6.62 |  24.25 |   5.84 |
| clifford T2 |   3.97 |  24.63 |   7.27 |
| s-curve     |   7.19 |  39.87 |   8.89 |
| saddle      |   5.95 |  41.48 |   6.77 |
| sphere S2   |   3.39 |  28.34 |   5.56 |
| sphere S3   |   3.48 |  41.81 |  15.49 |
| swiss roll  |   9.5  |  34.38 |   9.12 |
| torus R3    |   5.99 |  31.99 |  10.27 |
| trefoil     |   4.83 |  20.75 |   4.66 |

### Exp A, tuned per (manifold, noise) on ground truth, averaged over seeds: sigma=0.05

| manifold    |   lpca |   ldle |   lego |
|:------------|-------:|-------:|-------:|
| circle      |  12.89 |  33.3  |  10.54 |
| clifford T2 |   8.18 |  45.71 |  10.43 |
| s-curve     |  15.51 |  58.85 |  15.4  |
| saddle      |  14.17 |  61.07 |  12.27 |
| sphere S2   |   8.21 |  51.95 |   8.04 |
| sphere S3   |   7.79 |  60.06 |  15.45 |
| swiss roll  |  24.48 |  58.44 |  19.07 |
| torus R3    |  12.96 |  57.85 |  15.88 |
| trefoil     |  11.38 |  29.17 |   9.03 |

### Exp B (tubular noise), tuned per (manifold, noise) on ground truth, averaged over seeds

|                        |   lpca |   ldle |   lego |
|:-----------------------|-------:|-------:|-------:|
| ('clifford T2', 0.0)   |   2.47 |   2.7  |   7.21 |
| ('clifford T2', 0.005) |   2.55 |   8.44 |   7.2  |
| ('clifford T2', 0.01)  |   2.75 |  14.74 |   7.07 |
| ('clifford T2', 0.02)  |   3.44 |  24.33 |   6.99 |
| ('clifford T2', 0.03)  |   4.37 |  31.84 |   7.46 |
| ('clifford T2', 0.05)  |   6.59 |  45.17 |   9.17 |
| ('sphere S2', 0.0)     |   1.01 |   1.73 |   5.92 |
| ('sphere S2', 0.005)   |   1.25 |  10.01 |   5.81 |
| ('sphere S2', 0.01)    |   1.76 |  17.53 |   5.57 |
| ('sphere S2', 0.02)    |   3.06 |  28.47 |   5.42 |
| ('sphere S2', 0.03)    |   4.49 |  37.41 |   5.82 |
| ('sphere S2', 0.05)    |   7.47 |  52.27 |   7.19 |
| ('swiss roll', 0.0)    |   1.77 |   1.98 |  10.98 |
| ('swiss roll', 0.005)  |   3.66 |  13.26 |   9.35 |
| ('swiss roll', 0.01)   |   5.75 |  21.51 |   8.54 |
| ('swiss roll', 0.02)   |   9.23 |  34.48 |   8.86 |
| ('swiss roll', 0.03)   |  13.1  |  44    |  10.33 |
| ('swiss roll', 0.05)   |  24.36 |  58.47 |  17.46 |

### Exp C (sample size, sigma=0.02), tuned per (manifold, noise) on ground truth, averaged over seeds

|                       |   lpca |   ldle |   lego |
|:----------------------|-------:|-------:|-------:|
| ('clifford T2', 500)  |   6.3  |  41.47 |  18.38 |
| ('clifford T2', 1000) |   4.54 |  34.49 |  11.44 |
| ('clifford T2', 2000) |   3.97 |  24.63 |   7.27 |
| ('clifford T2', 4000) |   4.15 |  16.77 |   5.15 |
| ('sphere S2', 500)    |   3.16 |  45.45 |  13.32 |
| ('sphere S2', 1000)   |   2.93 |  38.99 |   7.77 |
| ('sphere S2', 2000)   |   3.39 |  28.34 |   5.56 |
| ('sphere S2', 4000)   |   4.32 |  19.76 |   3.89 |
| ('swiss roll', 500)   |  16.8  |  44.92 |  21.2  |
| ('swiss roll', 1000)  |  12.6  |  40.52 |  13.21 |
| ('swiss roll', 2000)  |   9.5  |  34.38 |   9.12 |
| ('swiss roll', 4000)  |   7.45 |  28.84 |   6.52 |

### Exp D (ambient dimension, fixed noise energy), tuned per (manifold, noise) on ground truth, averaged over seeds

|                     |   lpca |   ldle |   lego |
|:--------------------|-------:|-------:|-------:|
| ('sphere S2', 5)    |   4.05 |  24.33 |   5.99 |
| ('sphere S2', 10)   |   3.67 |  27.23 |   5.78 |
| ('sphere S2', 20)   |   3.41 |  28.53 |   5.56 |
| ('sphere S2', 50)   |   3.19 |  29.14 |   5.38 |
| ('sphere S2', 100)  |   3.1  |  29.48 |   5.34 |
| ('swiss roll', 5)   |  12.29 |  28.4  |   9.27 |
| ('swiss roll', 10)  |  10.24 |  32.94 |   9.21 |
| ('swiss roll', 20)  |   9.54 |  34.69 |   9.14 |
| ('swiss roll', 50)  |   8.99 |  35.94 |   8.96 |
| ('swiss roll', 100) |   8.83 |  36.16 |   8.86 |

### Win counts (tuned, Exp A, sigma>0)

|      |   count |
|:-----|--------:|
| lpca |      32 |
| lego |      13 |

### Median wall-clock per run (default config, n varies, 1 thread)

| method   |   time |
|:---------|-------:|
| ldle     |   3.48 |
| lego     |   3.51 |
| lpca     |   0.24 |

### Exp E (ambient N=1000, noise energy sigma*sqrt(N)), default

|                       |   lpca |   ldle |   lego |
|:----------------------|-------:|-------:|-------:|
| ('circle', 0.0)       |   0.12 |   0.11 |   3.65 |
| ('circle', 0.09)      |  44.91 |  51.12 |  17    |
| ('circle', 0.18)      |  53.68 |  58.57 |  28.79 |
| ('clifford T2', 0.0)  |   2.62 |   3.55 |   7.21 |
| ('clifford T2', 0.09) |  11.45 |  26.1  |   7.39 |
| ('clifford T2', 0.18) |  21.66 |  40.12 |   9.11 |
| ('s-curve', 0.0)      |   1.64 |   2.4  |  12.22 |
| ('s-curve', 0.09)     |  24.17 |  44.76 |  12.9  |
| ('s-curve', 0.18)     |  41.55 |  61.09 |  21.07 |
| ('saddle', 0.0)       |   1.11 |   1.25 |   3.94 |
| ('saddle', 0.09)      |  24.36 |  44.35 |  10.99 |
| ('saddle', 0.18)      |  41.66 |  61.61 |  18.57 |
| ('sphere S2', 0.0)    |   1.12 |   2.31 |   7.22 |
| ('sphere S2', 0.09)   |  13.91 |  30.94 |   6.97 |
| ('sphere S2', 0.18)   |  26.27 |  45.59 |   9.77 |
| ('sphere S3', 0.0)    |   2.59 |  16.79 |  23.07 |
| ('sphere S3', 0.09)   |   9.87 |  43.76 |  15.12 |
| ('sphere S3', 0.18)   |  18.84 |  56.8  |  13.85 |
| ('swiss roll', 0.0)   |   1.84 |   2.44 |  11.28 |
| ('swiss roll', 0.09)  |  19.85 |  39.06 |  11.25 |
| ('swiss roll', 0.18)  |  35.4  |  55.7  |  17.52 |
| ('torus R3', 0.0)     |   2.84 |   3.5  |  12.54 |
| ('torus R3', 0.09)    |  16.54 |  33.54 |  10    |
| ('torus R3', 0.18)    |  30.33 |  49    |  13.12 |
| ('trefoil', 0.0)      |   0.27 |   0.25 |   5.89 |
| ('trefoil', 0.09)     |  36.03 |  43.33 |  12.16 |
| ('trefoil', 0.18)     |  48.96 |  54.14 |  21.27 |

### Exp E (ambient N=1000, noise energy sigma*sqrt(N)), tuned

|                       |   lpca |   ldle |   lego |
|:----------------------|-------:|-------:|-------:|
| ('circle', 0.0)       |   0.1  |   0.07 |   1.42 |
| ('circle', 0.09)      |   6.75 |  32.76 |   6.19 |
| ('circle', 0.18)      |  13.24 |  46.51 |  11.5  |
| ('clifford T2', 0.0)  |   2.49 |   2.67 |   7.21 |
| ('clifford T2', 0.09) |   3.44 |  25.6  |   6.94 |
| ('clifford T2', 0.18) |   5.28 |  38.25 |   8.15 |
| ('s-curve', 0.0)      |   1.58 |   1.84 |  10.09 |
| ('s-curve', 0.09)     |   6.76 |  41.7  |   8.97 |
| ('s-curve', 0.18)     |  11.51 |  57.61 |  13.18 |
| ('saddle', 0.0)       |   1.1  |   0.93 |   3.94 |
| ('saddle', 0.09)      |   5.61 |  42.12 |   6.85 |
| ('saddle', 0.18)      |  10.78 |  58.56 |  10.49 |
| ('sphere S2', 0.0)    |   1.01 |   1.71 |   5.88 |
| ('sphere S2', 0.09)   |   3.03 |  29.38 |   5.35 |
| ('sphere S2', 0.18)   |   5.78 |  43.29 |   6.24 |
| ('sphere S3', 0.0)    |   1.77 |  14.25 |  20.15 |
| ('sphere S3', 0.09)   |   2.94 |  43.76 |  14.85 |
| ('sphere S3', 0.18)   |   5.01 |  56.19 |  13.57 |
| ('swiss roll', 0.0)   |   1.77 |   1.96 |  10.79 |
| ('swiss roll', 0.09)  |   8.69 |  36.06 |   8.86 |
| ('swiss roll', 0.18)  |  14.52 |  51.68 |  11.93 |
| ('torus R3', 0.0)     |   2.77 |   2.89 |  12.24 |
| ('torus R3', 0.09)    |   5.06 |  31.74 |   9.47 |
| ('torus R3', 0.18)    |   7.94 |  46.29 |  11.62 |
| ('trefoil', 0.0)      |   0.22 |   0.16 |   2.65 |
| ('trefoil', 0.09)     |   4.63 |  24.89 |   4.66 |
| ('trefoil', 0.18)     |   9.05 |  38.56 |   7.91 |

### Intrinsic-dimension accuracy (fraction of points with d_hat = d), N=20, all noise levels, by rule

| rule      |   lpca |   ldle |   lego |
|:----------|-------:|-------:|-------:|
| fvar=0.9  |  0.37  |  0.672 |  0.956 |
| fvar=0.95 |  0.281 |  0.562 |  0.848 |
| fvar=0.99 |  0.172 |  0.307 |  0.267 |
| ratio     |  0.704 |  0.209 |  0.591 |

### Intrinsic-dimension accuracy (fraction of points with d_hat = d), N=1000, by rule

| rule      |   lpca |   ldle |   lego |
|:----------|-------:|-------:|-------:|
| fvar=0.9  |  0.086 |  0.574 |  0.971 |
| fvar=0.95 |  0     |  0.342 |  0.919 |
| fvar=0.99 |  0     |  0.031 |  0.22  |
| ratio     |  0.995 |  0.096 |  0.798 |

### Dimension accuracy, fvar=0.9, N=20, by noise

|   sigma |   lpca |   ldle |   lego |
|--------:|-------:|-------:|-------:|
|   0     |  1     |  0.83  |  0.932 |
|   0.005 |  0.758 |  0.797 |  0.962 |
|   0.01  |  0.356 |  0.724 |  0.967 |
|   0.02  |  0.105 |  0.625 |  0.968 |
|   0.03  |  0.001 |  0.575 |  0.966 |
|   0.05  |  0     |  0.481 |  0.939 |

### Dimension accuracy, fvar=0.9, N=20, by manifold

| manifold    |   lpca |   ldle |   lego |
|:------------|-------:|-------:|-------:|
| circle      |  0.167 |  0.455 |  0.957 |
| clifford T2 |  0.498 |  0.818 |  0.996 |
| s-curve     |  0.326 |  0.733 |  0.928 |
| saddle      |  0.314 |  0.723 |  0.96  |
| sphere S2   |  0.467 |  0.789 |  0.996 |
| sphere S3   |  0.658 |  0.497 |  0.928 |
| swiss roll  |  0.355 |  0.763 |  0.923 |
| torus R3    |  0.377 |  0.764 |  0.933 |
| trefoil     |  0.167 |  0.506 |  0.979 |

### Speed (s, single thread, defaults, excl. kNN search) vs n (N=20)

|     n |   lpca |   ldle |   lego |
|------:|-------:|-------:|-------:|
|   500 |   0.04 |   0.24 |   0.08 |
|  1000 |   0.08 |   0.55 |   0.21 |
|  2000 |   0.16 |   1.6  |   0.92 |
|  4000 |   0.33 |   2.59 |   0.85 |
|  8000 |   0.67 |   5.81 |   1.94 |
| 16000 |   1.36 |  13.87 |   4.36 |

### Speed vs ambient N (n=2000)

|    N |   lpca |   ldle |   lego |
|-----:|-------:|-------:|-------:|
|   20 |   0.16 |   1.6  |   0.92 |
|   50 |   0.27 |   1.58 |   1    |
|  200 |   0.53 |   1.59 |   1.38 |
| 1000 |   2.01 |   1.68 |   3.56 |

### Accuracy (mean max angle, deg) in the speed runs vs n (N=20, sigma=0.02)

|     n |   lpca |   ldle |   lego |
|------:|-------:|-------:|-------:|
|   500 |  12.43 |  46.25 |  23.66 |
|  1000 |  12.81 |  41.83 |  16.14 |
|  2000 |  17.12 |  33.14 |   9.13 |
|  4000 |  23.75 |  33.56 |   6.51 |
|  8000 |  31.88 |  36.03 |   4.82 |
| 16000 |  40.16 |  38.37 |   3.73 |
