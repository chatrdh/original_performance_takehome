
from problem import myhash, HASH_STAGES

val = 2199965756
node_val = 112449971
xor_val = val ^ node_val
hashed = myhash(xor_val)
print(f"Val: {val}")
print(f"NodeVal: {node_val}")
print(f"Xor: {xor_val}")
print(f"Hashed: {hashed}")
print(f"Modulo 2: {hashed % 2}")

# Let's simulate my vector logic step by step to see where it diverges
def my_vector_logic_sim(a):
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }
    def r(x):
        return x % (2**32)
        
    for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        # My logic:
        # t1 = a op1 vc1
        # t2 = a op3 vc3
        # a = t1 op2 t2
        
        t1 = r(fns[op1](a, val1))
        t2 = r(fns[op3](a, val3))
        a_new = r(fns[op2](t1, t2))
        
        # Reference logic:
        # a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))
        # This is IDENTICAL order.
        
        print(f"Stage {i}: {a} -> {a_new}")
        a = a_new
    return a

print("Simulated Vector Hash:")
h2 = my_vector_logic_sim(xor_val)
print(f"H2: {h2}")
