# mchammer-pt architecture

This document is for developers working on mchammer-pt. For library
usage see the README.

## Mental model

mchammer-pt orchestrates parallel-tempering runs over an `mchammer`
canonical Monte Carlo or Wang-Landau ensemble. The runtime hierarchy
for Wang-Landau (REWL) is:

```
Pool   (SerialWangLandauPool / ProcessWangLandauPool)
 |     owns pool-level policy (flatness_mode, merge_cadence)
 |
 +-- Slot   (WangLandauSlot Protocol)
 |   |     covers one energy window;
 |   |     carries walker_states + schedule + flatness_limit + phase;
 |   |     applies the CoordinatorPlan returned by decide_block_actions
 |   |
 |   +-- If W=1: WangLandauReplica IS the slot (slot and walker collapse)
 |   |
 |   +-- If W>1: WangLandauWindowGroup IS the slot
 |       +-- contains N WangLandauReplicas (the walkers)
 |
 +-- (Each WangLandauReplica wraps a CoordinatedWangLandauEnsemble
      from mchammer.)
```

A **slot** is the unit that gets one `CoordinatorPlan` per block. A
slot covers exactly one energy window. Walkers live inside the slot
when there is more than one. For `n_walkers_per_window == 1`, the slot
IS the walker (a bare `WangLandauReplica`). For `n_walkers_per_window
> 1`, the slot is a `WangLandauWindowGroup` containing N
`WangLandauReplica` walkers that share the window.

"Window" is not a layer in the hierarchy; it is a property of every
slot (`slot.energy_window`).

The Pool drives every slot uniformly through the `WangLandauSlot`
Protocol. There is no `isinstance` dispatch at the pool level -- both
slot kinds satisfy the same contract.

## The four-phase pipeline

Both `SerialWangLandauPool.advance_all` and
`ProcessWangLandauPool.advance_all` follow the same shape:

1. **Advance.** Every walker runs `n_steps` MC steps. The serial
   pool loops over slots in-process. The process pool fans out one
   batched IPC round across every walker in every slot.
2. **Collect.** Each slot's `walker_states: Sequence[WalkerPostBlockState]`
   is populated. For W=1 it is a 1-tuple; for W>1 it contains
   one snapshot per walker.
3. **Decide.** The pool builds a `SlotView` for each slot -- reading
   `walker_states`, `phase`, `schedule`, `flatness_limit` off the
   slot, and `flatness_mode`, `merge_cadence` off the pool -- and
   calls the pure function `decide_block_actions(view) ->
   CoordinatorPlan`.
4. **Apply.** Each pool applies the per-slot plans in a
   backend-appropriate way. The serial pool calls `slot.apply_plan(plan)`
   per slot. The process pool keeps its existing three batched IPC
   rounds (FORCE_HALVE / SET_ENTROPY / SET_PHASE), each parallelised
   across walkers of all slots that need the action.

After apply, both pools re-roll the exchange-walker index on each slot
(no-op for W=1 slots; meaningful for W>1).

## Policy lives at the pool

`flatness_mode` and `merge_cadence` are orchestrator-level parameters:
they apply uniformly to all windows in a run. They live on the pool,
not on individual slots. Slots carry their own state (`walker_states`,
`schedule`, `flatness_limit`, `phase`) but not the policy that decides
when collective halving fires.

The pool's `_view_of(slot)` adapter combines slot state with pool
policy to build the `SlotView` consumed by `decide_block_actions`.

## The coordinator pure function

`decide_block_actions(view: SlotView) -> CoordinatorPlan` is the
single home for the collective-halving policy. It is pure -- same
input, same output, no side effects. Direct unit tests in
`tests/test_wl_coordinator.py` exercise the policy matrix
(`flatness_mode` x `merge_cadence` x `schedule` x `phase`).

The decision body covers:

- **Phase gate.** In the 1/t phase the plan is empty (no halve, no
  merge, no switch).
- **Halve gate.** In the halving phase the plan halves if either
  `flatness_mode="per_walker"` and every walker is independently
  flat, or `flatness_mode="pooled"` and the summed histogram across
  walkers is flat.
- **Merge.** If halving and `merge_cadence="at_halve"` and W>1,
  the plan includes merged entropy for write-back.
- **BP switch.** If halving and `schedule="1_over_t"` and every
  walker satisfies `1/t > f_post_halve`, the plan flips the phase
  to `"1_over_t"`.

## Serial vs process

Both backends follow the four-phase shape. The differences are
mechanical:

- **Serial** runs MC in-process and applies plans by calling
  `slot.apply_plan` for each slot. No IPC; no batching.
- **Process** runs MC in worker subprocesses (one per walker) and
  applies plans by three batched IPC rounds (FORCE_HALVE,
  SET_ENTROPY, SET_PHASE), each parallelised across all walkers of
  all slots that need the corresponding action.

The shared shape makes the unification visible at the call site:
`SerialWangLandauPool.advance_all` and `ProcessWangLandauPool.advance_all`
read structurally as the same algorithm, with backend-specific
implementations of the advance and apply primitives.

## Where to look

- `mchammer_pt/wl_coordinator.py` -- the policy types
  (`WalkerPostBlockState`, `SlotView`, `CoordinatorPlan`,
  `FlatnessMode`, `MergeCadence`) and the pure function
  `decide_block_actions`.
- `mchammer_pt/wl_replica.py` -- `WangLandauReplica` (one walker;
  also serves as the 1-walker slot) and the `WangLandauSlot`
  Protocol.
- `mchammer_pt/wl_window_group.py` -- `WangLandauWindowGroup`
  (multi-walker slot).
- `mchammer_pt/parallel/serial.py` -- `SerialWangLandauPool`.
- `mchammer_pt/parallel/processes.py` -- `ProcessWangLandauPool`.
- `mchammer_pt/wl.py` -- `WangLandauParallelTempering`
  orchestrator.
