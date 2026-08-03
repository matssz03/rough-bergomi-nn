# Table 3 — Binomial convergence to Black-Scholes

Maximum absolute pricing error across the strike-maturity grid of Section 3.2 (8 maturities x 11 strikes = 88 contracts), for European call options at S = 100, r = 0%, sigma = 20%, using the CRR and JR schemes at three values of the number of time steps N.

| Scheme | N = 128 | N = 256 | N = 512 | Rate 128->256 | Rate 256->512 |
|---|---|---|---|---|---|
| CRR | 0.02194 | 0.01098 | 0.00549 | 1.00 | 1.00 |
| JR | 0.01898 | 0.00836 | 0.00490 | 1.18 | 0.77 |

*Empirical convergence rates are computed as $\log_2(\text{err}(N) / \text{err}(2N))$. A rate close to 1 confirms the theoretical $O(1/N)$ convergence of both binomial schemes to the Black-Scholes reference. At N = 512, the maximum absolute error falls below one basis point on the ATM one-year contract (BS reference price: 7.5581).*
