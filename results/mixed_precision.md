#### Forward / Backward timing

| model_size | batch_size | context_length | vocab_size | amp | device | phase | mean_ms | std_ms | tok_s | num_warmups | num_measures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small | 4 | 256 | 10000 | none | cpu | forward | 538.088 | 227.580 | 1903.035 | 5 | 10 |
| small | 4 | 256 | 10000 | none | cpu | backward | 791.492 | 135.817 | 1293.759 | 5 | 10 |
| small | 4 | 256 | 10000 | none | cpu | optimizer | 303.629 | 74.662 | 3372.540 | 5 | 10 |
| small | 4 | 256 | 10000 | bf16 | cpu | forward | 458.358 | 54.240 | 2234.062 | 5 | 10 |
| small | 4 | 256 | 10000 | bf16 | cpu | backward | 746.608 | 47.482 | 1371.537 | 5 | 10 |
| small | 4 | 256 | 10000 | bf16 | cpu | optimizer | 243.848 | 17.951 | 4199.339 | 5 | 10 |
| small | 4 | 256 | 10000 | none | cpu | forward | 527.331 | 79.501 | 1941.856 | 5 | 10 |
| small | 4 | 256 | 10000 | none | cpu | backward | 792.154 | 57.957 | 1292.678 | 5 | 10 |
| small | 4 | 256 | 10000 | none | cpu | optimizer | 275.597 | 35.441 | 3715.568 | 5 | 10 |
| small | 4 | 256 | 10000 | bf16 | cpu | forward | 497.676 | 81.772 | 2057.564 | 5 | 10 |
| small | 4 | 256 | 10000 | bf16 | cpu | backward | 769.917 | 68.197 | 1330.014 | 5 | 10 |
| small | 4 | 256 | 10000 | bf16 | cpu | optimizer | 273.335 | 26.807 | 3746.318 | 5 | 10 |
| medium | 4 | 256 | 10000 | none | cpu | forward | 25322.9 | 3139.603 | 40.438 | 5 | 10 |
| medium | 4 | 256 | 10000 | none | cpu | backward | 10665.0 | 2862.987 | 96.015 | 5 | 10 |
| medium | 4 | 256 | 10000 | none | cpu | optimizer | 23926.3 | 3817.090 | 42.798 | 5 | 10 |
| medium | 4 | 256 | 10000 | bf16 | cpu | forward | 19768.5 | 1665.078 | 51.800 | 5 | 10 |
| medium | 4 | 256 | 10000 | bf16 | cpu | backward | 7573.201 | 1621.160 | 135.214 | 5 | 10 |
| medium | 4 | 256 | 10000 | bf16 | cpu | optimizer | 17928.0 | 1351.143 | 57.117 | 5 | 10 |
