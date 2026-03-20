"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

import random
import unittest

from problem import (
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    """
    Builds optimized VLIW SIMD kernels for the tree traversal hash algorithm.
    """

    UNROLL = 16
    BLOCK_WIDTH = 4
    BLOCK_STARTS = tuple(range(0, UNROLL, BLOCK_WIDTH))
    BLOCK_CYCLES = VLEN * BLOCK_WIDTH // SLOT_LIMITS["load"]

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.pending_const_loads = []
        
    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def emit(self, instr):
        """Emit a single VLIW instruction bundle."""
        self.instrs.append(instr)

    def alloc_vector_bank(self, prefix):
        return [self.alloc_scratch(f"{prefix}{i}", VLEN) for i in range(self.UNROLL)]

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        """Allocate a constant scratch slot, batching the loads later."""
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.const_map[val] = addr
            self.pending_const_loads.append((addr, val))
        return self.const_map[val]

    def flush_const_loads(self):
        while self.pending_const_loads:
            slots = [
                ("const", addr, val)
                for addr, val in self.pending_const_loads[: SLOT_LIMITS["load"]]
            ]
            self.emit({"load": slots})
            del self.pending_const_loads[: SLOT_LIMITS["load"]]

    def hash_pre_slots(self, pair_start, stage_i, v_val, v_tmp1, v_tmp2, hash_vconsts):
        op1, _, _, op3, _ = HASH_STAGES[stage_i]
        vc1, vc3 = hash_vconsts[stage_i]
        return [
            (op1, v_tmp1[pair_start], v_val[pair_start], vc1),
            (op3, v_tmp2[pair_start], v_val[pair_start], vc3),
            (op1, v_tmp1[pair_start + 1], v_val[pair_start + 1], vc1),
            (op3, v_tmp2[pair_start + 1], v_val[pair_start + 1], vc3),
        ]

    def hash_combine_slots(self, pair_start, stage_i, v_val, v_tmp1, v_tmp2):
        _, _, op2, _, _ = HASH_STAGES[stage_i]
        return [
            (op2, v_val[pair_start], v_tmp1[pair_start], v_tmp2[pair_start]),
            (op2, v_val[pair_start + 1], v_tmp1[pair_start + 1], v_tmp2[pair_start + 1]),
        ]

    def block_load_schedule(self, block_start, v_node_val, v_addr):
        schedule = []
        for k in range(0, VLEN, 2):
            for u in range(block_start, block_start + self.BLOCK_WIDTH):
                schedule.append([
                    ("load", v_node_val[u] + k, v_addr[u] + k),
                    ("load", v_node_val[u] + k + 1, v_addr[u] + k + 1),
                ])
        assert len(schedule) == self.BLOCK_CYCLES
        return schedule

    def block_hash_schedule(self, block_start, v_val, v_node_val, v_tmp1, v_tmp2, hash_vconsts):
        pair0 = block_start
        pair1 = block_start + 2
        last_stage = len(HASH_STAGES) - 1
        schedule = [[
            ("^", v_val[block_start], v_val[block_start], v_node_val[block_start]),
            ("^", v_val[block_start + 1], v_val[block_start + 1], v_node_val[block_start + 1]),
            ("^", v_val[block_start + 2], v_val[block_start + 2], v_node_val[block_start + 2]),
            ("^", v_val[block_start + 3], v_val[block_start + 3], v_node_val[block_start + 3]),
        ]]
        schedule.append(self.hash_pre_slots(pair0, 0, v_val, v_tmp1, v_tmp2, hash_vconsts))
        for stage_i in range(len(HASH_STAGES)):
            schedule.append(
                self.hash_combine_slots(pair0, stage_i, v_val, v_tmp1, v_tmp2)
                + self.hash_pre_slots(pair1, stage_i, v_val, v_tmp1, v_tmp2, hash_vconsts)
            )
            if stage_i < last_stage:
                schedule.append(
                    self.hash_combine_slots(pair1, stage_i, v_val, v_tmp1, v_tmp2)
                    + self.hash_pre_slots(pair0, stage_i + 1, v_val, v_tmp1, v_tmp2, hash_vconsts)
                )
        schedule.append(self.hash_combine_slots(pair1, last_stage, v_val, v_tmp1, v_tmp2))
        assert len(schedule) == 14
        return schedule

    def block_idx_update_schedule(
        self,
        block_start,
        wraps_after_round,
        v_idx,
        v_val,
        v_tmp1,
        one,
    ):
        schedule = [[] for _ in range(self.BLOCK_CYCLES)]
        lanes = [
            (u, lane)
            for u in range(block_start, block_start + self.BLOCK_WIDTH)
            for lane in range(VLEN)
        ]

        if wraps_after_round:
            zero_ops = [
                ("^", v_idx[u] + lane, v_idx[u] + lane, v_idx[u] + lane)
                for u, lane in lanes
            ]
            schedule[0] = zero_ops[:12]
            schedule[1] = zero_ops[12:24]
            schedule[2] = zero_ops[24:]
            return schedule

        phases = [
            [("&", v_tmp1[u] + lane, v_val[u] + lane, one) for u, lane in lanes],
            [("+", v_idx[u] + lane, v_idx[u] + lane, v_idx[u] + lane) for u, lane in lanes],
            [("+", v_idx[u] + lane, v_idx[u] + lane, v_tmp1[u] + lane) for u, lane in lanes],
            [("+", v_idx[u] + lane, v_idx[u] + lane, one) for u, lane in lanes],
        ]

        for phase_i, ops in enumerate(phases):
            cycle_i = phase_i * 3
            schedule[cycle_i] = ops[:12]
            schedule[cycle_i + 1] = ops[12:24]
            schedule[cycle_i + 2] = ops[24:]
        return schedule

    def block_addr_prepare_schedule(self, block_start, v_idx, v_addr, forest_values_p):
        schedule = [[] for _ in range(self.BLOCK_CYCLES)]
        ops = [
            ("+", v_addr[u] + lane, v_idx[u] + lane, forest_values_p)
            for u in range(block_start, block_start + self.BLOCK_WIDTH)
            for lane in range(VLEN)
        ]
        schedule[12] = ops[:8]
        schedule[13] = ops[8:16]
        schedule[14] = ops[16:24]
        schedule[15] = ops[24:]
        return schedule

    def emit_block_load(self, block_start, v_node_val, v_addr):
        for load_slots in self.block_load_schedule(block_start, v_node_val, v_addr):
            self.emit({"load": load_slots})

    def emit_task_segment(
        self,
        block_start,
        next_block_start,
        update_block_start,
        update_wraps_after_round,
        addr_block_start,
        v_idx,
        v_val,
        v_node_val,
        v_tmp1,
        v_tmp2,
        v_addr,
        one,
        forest_values_p,
        hash_vconsts,
    ):
        valu_schedule = self.block_hash_schedule(
            block_start, v_val, v_node_val, v_tmp1, v_tmp2, hash_vconsts
        )

        load_schedule = (
            self.block_load_schedule(next_block_start, v_node_val, v_addr)
            if next_block_start is not None
            else [[] for _ in range(self.BLOCK_CYCLES)]
        )

        alu_schedule = [[] for _ in range(self.BLOCK_CYCLES)]
        if update_block_start is not None:
            alu_schedule = self.block_idx_update_schedule(
                update_block_start,
                update_wraps_after_round,
                v_idx,
                v_val,
                v_tmp1,
                one,
            )
        if addr_block_start is not None:
            addr_schedule = self.block_addr_prepare_schedule(
                addr_block_start, v_idx, v_addr, forest_values_p
            )
            for cycle_i in range(self.BLOCK_CYCLES):
                alu_schedule[cycle_i].extend(addr_schedule[cycle_i])

        for cycle_i in range(self.BLOCK_CYCLES):
            instr = {}
            if load_schedule[cycle_i]:
                instr["load"] = load_schedule[cycle_i]
            if cycle_i < len(valu_schedule):
                instr["valu"] = valu_schedule[cycle_i]
            if alu_schedule[cycle_i]:
                instr["alu"] = alu_schedule[cycle_i]
            if instr:
                self.emit(instr)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel with 16x batch unrolling and a software pipeline that
        overlaps the next block's gathers with the current block's hash.
        """
        assert batch_size % (self.UNROLL * VLEN) == 0

        # === PHASE 1: Load constants and pointers ===
        one = self.scratch_const(1, "one")

        inp_values_p = self.alloc_scratch("inp_values_p")
        forest_values_p = self.alloc_scratch("forest_values_p")
        batch_size_reg = self.alloc_scratch("batch_size_reg")

        pointer_offsets = {
            "forest_values": self.scratch_const(4),
            "inp_values": self.scratch_const(6),
            "batch_size": self.scratch_const(2),
        }

        # === PHASE 2: Allocate runtime constants ===
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        hash_vconsts = []
        hash_broadcasts = []
        for stage_i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1)
            vc1 = self.alloc_scratch(f"hc{stage_i}_1", VLEN)
            c3 = self.scratch_const(val3)
            vc3 = self.alloc_scratch(f"hc{stage_i}_3", VLEN)
            hash_broadcasts.extend([
                ("vbroadcast", vc1, c1),
                ("vbroadcast", vc3, c3),
            ])
            hash_vconsts.append((vc1, vc3))

        # === PHASE 3: Allocate registers for block-pipelined execution ===
        v_idx = self.alloc_vector_bank("v_idx")
        v_val = self.alloc_vector_bank("v_val")
        v_node_val = self.alloc_vector_bank("v_node_val")
        v_tmp1 = self.alloc_vector_bank("v_tmp1_")
        v_tmp2 = self.alloc_vector_bank("v_tmp2_")
        v_addr = self.alloc_vector_bank("v_addr")

        batch_base_vals = [
            [self.alloc_scratch(f"batch_base{buf}_val{u}") for u in range(self.UNROLL)]
            for buf in range(2)
        ]
        offset_consts = [self.scratch_const(u * VLEN) for u in range(self.UNROLL)]
        batch_base_ptr = self.alloc_scratch("batch_base_ptr")
        tasks = [(r, block_start) for r in range(rounds) for block_start in self.BLOCK_STARTS]
        n_batches = batch_size // (self.UNROLL * VLEN)
        batch_step = self.UNROLL * VLEN
        batch_offset_consts = {
            batch_i: self.scratch_const(batch_i * batch_step)
            for batch_i in range(1, n_batches)
        }

        self.flush_const_loads()

        self.emit({"load": [
            ("load", forest_values_p, pointer_offsets["forest_values"]),
        ]})
        self.emit({"load": [
            ("load", inp_values_p, pointer_offsets["inp_values"]),
            ("load", batch_size_reg, pointer_offsets["batch_size"]),
        ]})
        self.emit({"valu": [
            ("vbroadcast", v_forest_p, forest_values_p),
        ]})
        for i in range(0, len(hash_broadcasts), SLOT_LIMITS["valu"]):
            self.emit({"valu": hash_broadcasts[i : i + SLOT_LIMITS["valu"]]})

        pending_store_chunks = None
        for batch_i in range(n_batches):
            batch_base_val = batch_base_vals[batch_i % 2]
            base_source = inp_values_p
            overlapped_store_i = 0
            if batch_i != 0:
                instr = {"alu": [("+", batch_base_ptr, inp_values_p, batch_offset_consts[batch_i])]}
                if pending_store_chunks is not None:
                    instr["store"] = pending_store_chunks[0]
                    overlapped_store_i = 1
                self.emit(instr)
                base_source = batch_base_ptr

            base_ops = [
                ("+", batch_base_val[u], base_source, offset_consts[u])
                for u in range(self.UNROLL)
            ]

            zero_idx_ops = [("^", v_idx[u], v_idx[u], v_idx[u]) for u in range(self.UNROLL)]
            seed_addr_ops = [("+", v_addr[u], v_forest_p, v_idx[u]) for u in range(self.UNROLL)]
            base_chunks = [base_ops[:12], base_ops[12:]]
            valu_chunks = [
                zero_idx_ops[:6],
                zero_idx_ops[6:12],
                zero_idx_ops[12:],
                seed_addr_ops[:6],
                seed_addr_ops[6:12],
                seed_addr_ops[12:],
            ]
            vload_chunks = [
                [
                    ("vload", v_val[u], batch_base_val[u]),
                    ("vload", v_val[u + 1], batch_base_val[u + 1]),
                ]
                for u in range(0, self.UNROLL, 2)
            ]

            # The valu setup and value loads are independent once each address scalar
            # is ready, so overlap them instead of running them as separate phases.
            for cycle_i in range(max(len(base_chunks), len(valu_chunks), len(vload_chunks) + 1)):
                instr = {}
                if cycle_i < len(base_chunks) and base_chunks[cycle_i]:
                    instr["alu"] = base_chunks[cycle_i]
                if cycle_i < len(valu_chunks) and valu_chunks[cycle_i]:
                    instr["valu"] = valu_chunks[cycle_i]
                if cycle_i > 0:
                    load_i = cycle_i - 1
                    if load_i < len(vload_chunks):
                        instr["load"] = vload_chunks[load_i]
                if pending_store_chunks is not None:
                    store_i = overlapped_store_i + cycle_i
                    if store_i < len(pending_store_chunks):
                        instr["store"] = pending_store_chunks[store_i]
                self.emit(instr)

            # === PHASE 5: Unrolled rounds as a block pipeline ===
            self.emit_block_load(tasks[0][1], v_node_val, v_addr)
            for task_i, (round_i, block_start) in enumerate(tasks):
                next_block_start = tasks[task_i + 1][1] if task_i + 1 < len(tasks) else None
                update_block_start = None
                update_wraps_after_round = False
                update_task_i = task_i - 1
                if update_task_i >= 0 and update_task_i + len(self.BLOCK_STARTS) < len(tasks):
                    update_block_start = tasks[update_task_i][1]
                    update_round_i = tasks[update_task_i][0]
                    update_wraps_after_round = (update_round_i + 1) % (forest_height + 1) == 0

                addr_block_start = None
                if task_i + 2 < len(tasks) and task_i + 2 >= len(self.BLOCK_STARTS):
                    addr_block_start = tasks[task_i + 2][1]

                self.emit_task_segment(
                    block_start=block_start,
                    next_block_start=next_block_start,
                    update_block_start=update_block_start,
                    update_wraps_after_round=update_wraps_after_round,
                    addr_block_start=addr_block_start,
                    v_idx=v_idx,
                    v_val=v_val,
                    v_node_val=v_node_val,
                    v_tmp1=v_tmp1,
                    v_tmp2=v_tmp2,
                    v_addr=v_addr,
                    one=one,
                    forest_values_p=forest_values_p,
                    hash_vconsts=hash_vconsts,
                )

            pending_store_chunks = [
                [
                    ("vstore", batch_base_val[u], v_val[u]),
                    ("vstore", batch_base_val[u + 1], v_val[u + 1]),
                ]
                for u in range(0, self.UNROLL, 2)
            ]

        for store_i, store_slots in enumerate(pending_store_chunks):
            instr = {"store": store_slots}
            if store_i == len(pending_store_chunks) - 1:
                instr["flow"] = [("pause",)]
            self.emit(instr)

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.prints = prints
    machine.run()
    for ref_mem in reference_kernel2(mem, value_trace):
        pass

    inp_values_p = ref_mem[6]
    if prints:
        print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect output values"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
