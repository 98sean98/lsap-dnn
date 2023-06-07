data file naming convention:

```text
data_{n}_{bound}_{size}.h5
```

where

- `n` is the number of jobs / people
- `bound` is the upper bound of the cost for each job-person pair, i.e. 0 <= `c_ij` < `bound`
- `size` is the sample size
