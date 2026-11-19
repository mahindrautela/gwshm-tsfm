## within-specimen

|                                                          |   mean |   std |   auc_measurement |   detected_zero_fa |
|:---------------------------------------------------------|-------:|------:|------------------:|-------------------:|
| ('Chronos-2', 'PCA+ocSVM')                               |  0.513 | 0.045 |             0.488 |               0.25 |
| ('Chronos-2', 'kNN distance')                            |  0.825 | 0.122 |             0.925 |               0.8  |
| ('Chronos-2', 'same-path distance')                      |  0.882 | 0.041 |             1     |               1    |
| ('Chronos-Bolt (base)', 'PCA+ocSVM')                     |  0.598 | 0.026 |             1     |               1    |
| ('Chronos-Bolt (base)', 'kNN distance')                  |  0.94  | 0.073 |             1     |               1    |
| ('Chronos-Bolt (base)', 'same-path distance')            |  0.979 | 0.018 |             1     |               1    |
| ('MOMENT-1 (base)', 'PCA+ocSVM')                         |  0.624 | 0.054 |             0.962 |               0.95 |
| ('MOMENT-1 (base)', 'kNN distance')                      |  0.929 | 0.075 |             1     |               1    |
| ('MOMENT-1 (base)', 'same-path distance')                |  0.954 | 0.051 |             1     |               1    |
| ('Mantis-8M', 'PCA+ocSVM')                               |  0.564 | 0.065 |             0.95  |               0.95 |
| ('Mantis-8M', 'kNN distance')                            |  0.962 | 0.082 |             1     |               1    |
| ('Mantis-8M', 'same-path distance')                      |  0.979 | 0.057 |             1     |               1    |
| ('Raw signal', 'PCA+ocSVM')                              |  0.226 | 0.098 |             0.1   |               0    |
| ('Raw signal', 'baseline subtraction (1 - correlation)') |  0.966 | 0.048 |             1     |               1    |
| ('Raw signal', 'baseline subtraction (residual energy)') |  0.914 | 0.119 |             0.95  |               0.95 |
| ('Raw signal', 'kNN distance')                           |  0.824 | 0.173 |             0.888 |               0.8  |
| ('Raw signal', 'same-path distance')                     |  0.91  | 0.125 |             0.95  |               0.95 |
| ('TTM-R2', 'PCA+ocSVM')                                  |  0.436 | 0.109 |             0.25  |               0.25 |
| ('TTM-R2', 'kNN distance')                               |  0.955 | 0.062 |             1     |               1    |
| ('TTM-R2', 'same-path distance')                         |  0.971 | 0.034 |             1     |               1    |
| ('TiRex', 'PCA+ocSVM')                                   |  0.602 | 0.041 |             1     |               1    |
| ('TiRex', 'kNN distance')                                |  0.909 | 0.08  |             1     |               1    |
| ('TiRex', 'same-path distance')                          |  0.978 | 0.021 |             1     |               1    |
| ('|FFT| spectrum', 'PCA+ocSVM')                          |  0.472 | 0.121 |             0.275 |               0.15 |
| ('|FFT| spectrum', 'kNN distance')                       |  0.487 | 0.167 |             0.375 |               0.2  |
| ('|FFT| spectrum', 'same-path distance')                 |  0.721 | 0.094 |             0.95  |               0.95 |

## cross-specimen

|                                                          |   mean |   std |   auc_measurement |   detected_zero_fa |
|:---------------------------------------------------------|-------:|------:|------------------:|-------------------:|
| ('Chronos-2', 'PCA+ocSVM')                               |  0.54  | 0.05  |             0.75  |               0.5  |
| ('Chronos-2', 'kNN distance')                            |  0.469 | 0.048 |             0.219 |               0    |
| ('Chronos-2', 'same-path distance')                      |  0.537 | 0.062 |             0.688 |               0.5  |
| ('Chronos-Bolt (base)', 'PCA+ocSVM')                     |  0.585 | 0.06  |             0.875 |               0.75 |
| ('Chronos-Bolt (base)', 'kNN distance')                  |  0.611 | 0.13  |             0.75  |               0.75 |
| ('Chronos-Bolt (base)', 'same-path distance')            |  0.669 | 0.074 |             1     |               1    |
| ('MOMENT-1 (base)', 'PCA+ocSVM')                         |  0.468 | 0.121 |             0.281 |               0.25 |
| ('MOMENT-1 (base)', 'kNN distance')                      |  0.553 | 0.148 |             0.75  |               0.75 |
| ('MOMENT-1 (base)', 'same-path distance')                |  0.472 | 0.11  |             0.25  |               0.25 |
| ('Mantis-8M', 'PCA+ocSVM')                               |  0.581 | 0.103 |             0.781 |               0.75 |
| ('Mantis-8M', 'kNN distance')                            |  0.717 | 0.188 |             0.75  |               0.75 |
| ('Mantis-8M', 'same-path distance')                      |  0.813 | 0.05  |             1     |               1    |
| ('Raw signal', 'PCA+ocSVM')                              |  0.276 | 0.128 |             0.125 |               0    |
| ('Raw signal', 'baseline subtraction (1 - correlation)') |  0.566 | 0.124 |             0.75  |               0.75 |
| ('Raw signal', 'baseline subtraction (residual energy)') |  0.194 | 0.054 |             0     |               0    |
| ('Raw signal', 'kNN distance')                           |  0.152 | 0.16  |             0.125 |               0    |
| ('Raw signal', 'same-path distance')                     |  0.236 | 0.161 |             0.125 |               0    |
| ('TTM-R2', 'PCA+ocSVM')                                  |  0.448 | 0.041 |             0.188 |               0    |
| ('TTM-R2', 'kNN distance')                               |  0.636 | 0.111 |             0.812 |               0.75 |
| ('TTM-R2', 'same-path distance')                         |  0.506 | 0.064 |             0.5   |               0.5  |
| ('TiRex', 'PCA+ocSVM')                                   |  0.598 | 0.116 |             0.75  |               0.75 |
| ('TiRex', 'kNN distance')                                |  0.626 | 0.107 |             0.781 |               0.75 |
| ('TiRex', 'same-path distance')                          |  0.661 | 0.098 |             0.844 |               0.75 |
| ('|FFT| spectrum', 'PCA+ocSVM')                          |  0.543 | 0.059 |             0.531 |               0    |
| ('|FFT| spectrum', 'kNN distance')                       |  0.127 | 0.116 |             0     |               0    |
| ('|FFT| spectrum', 'same-path distance')                 |  0.556 | 0.054 |             0.625 |               0.5  |

