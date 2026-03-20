
from perf_takehome import do_kernel_test

print("Running small test with unroll factor...")
# Must use batch_size >= 32 for UNROLL=4
cycles = do_kernel_test(forest_height=4, rounds=1, batch_size=32, trace=False, prints=False)
print(f"Cycles: {cycles}")
