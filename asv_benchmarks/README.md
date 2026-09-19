# Graphix benchmarks

Benchmarking Graphix with [Airspeed Velocity](https://asv.readthedocs.io/en/stable/index.html).

## Basic usage

Ensure _airspeed velocity_ is installed:

```
pip install asv
```

To benchmark the last commit run
```
asv run
```
on the `asv_benchmarks` folder.

To visualize the results run
```
asv publish
asv preview
```


## Further options

Airspeed velocity allows comparing over various commits. For instance, to benchmark the last `n = 3` commits run:

```
git log --format="%H" -n 3 >  hashes_to_benchmark.txt 
asv run HASHFILE:hashes_to_benchmark.txt
```

See the [user reference](https://asv.readthedocs.io/en/stable/user_reference.html) for additional information.
