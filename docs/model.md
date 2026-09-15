# Model reference

## Allocation methods

`Simulator` uses `mode="gscbba"` by default. Each harvester builds a bundle from
candidate rows and exchanges ownership records with its neighbors. Set
`spatial=True` to use KD-tree candidate filtering. With full connectivity,
`balancing=True` assigns free-row length in proportion to machine speed.

The proximity rule limits how far a bundle can grow across the field, but it can
leave gaps. Re-sorting rows can change their entry directions and increase a
stored bid. The algorithm has no established diminishing-marginal-gain or
finite-round convergence guarantee.

When the radio graph splits, `component_balancing=True` lets each connected
subteam split the rows it already holds in proportion to speed. The subteam
accepts the new split only if its projected finish time, including the remaining
current row, falls by more than 0.1%. Balancing writes the whole partition into
every member's records, so stale claims cannot later release a new owner's rows.
`idle_help=True` lets an idle machine trigger an allocation cycle in its own
component when its view of the field has changed.

`mode="marginal"` uses endpoint insertion, capped marginal bids, and per-row
last-writer-wins records. It is retained for regression tests only.

`gscbba.standard_cbba` implements textbook CBBA (Choi, Brunet, and How, 2009).
Each agent scores a bundle with a time-discounted reward, inserts every new row
at its best path position, and resolves conflicts with the full
update/reset/leave decision table and per-agent timestamps. Rows after a lost
row are released. The benchmark uses a discount of 0.9 per mean row time and a
bundle limit of ceil(1.5 M/N), chosen on seeds 900 to 902, which are not
benchmark seeds.

`optimal_contiguous_partition` evaluates position-ordered contiguous cuts with
a fixed ascending sweep. It is exact within that continuous-cost route class.
The benchmark executes its assignment with the same 0.5 s motion steps as the
auction methods, so the executed makespan is not a certified discrete-time
optimum. `static_speed_split` computes a speed-proportional allocation once.

`milp_assignment` optimizes ownership and row order under the nearest-endpoint
entry rule. Both PuLP status fields must certify optimality; a time-limited
incumbent is rejected. The solver supports at most 30 rows and four machines.
The cached certification experiment contains three six-row instances.

## Events and information

Rows and initial machine states are known before planning. Future events are not
supplied to any planner.

With `information="shared"` (the default), candidate queries use shared
completion state and scripted failures are announced globally after
`loss_detect_delay`. Balancing uses the position and speed of the connected team.

With `information="local"`, no machine reads the global completion set. Every
`comm_period` seconds, machines in one radio component gossip their completed
rows, positions, speeds, remaining waypoints, failure beliefs, and the time they
last heard every peer. A machine declares a peer failed when it has not heard
the peer for `heartbeat_timeout` seconds and the peer's dead-reckoned position
lies within `detect_fraction * comm_range`. A peer whose reported plan has run
out is never suspected. A declared failure releases the peer's claims locally
and spreads by gossip. A machine that later hears the peer again withdraws the
belief. When an idle machine has not heard an owner for `claim_ttl` seconds, it
may treat that owner's claims as expired. A joining machine starts with no
completion knowledge and is known only to its neighbors. False detections,
detection delays, and duplicate services are recorded.

Range-limited edges constrain bid exchanges. `packet_loss` drops each record
transmission and gossip broadcast independently. `round_latency` and
`latency_jitter` add delay to every forwarding hop of an allocation cycle, and
idle machines wait for that delay before acting on a new plan. The balancing
exchange uses acknowledged retransmission, so loss adds delay rather than
inconsistency. The network model does not serialize bytes or emulate a specific
transport stack.

`allocation_policy="centralized"` is an oracle: it re-solves the fixed-sweep DP
with global, current information at every event. `allocation_policy="centralized_base"`
places a coordinator at `base_position`. The coordinator reaches only machines
in its radio component, possibly over several hops. It knows what those
machines report under the chosen information mode. Unreachable machines keep
their last plan and drive back toward the base when idle. Plan delivery costs
two latency periods plus retransmissions.

Centralized and auction-based replanning preserve each active machine's current
row. The centralized planner accounts for remaining service and any initial
turn allowance before allocating the free rows.

## Motion and turn costs

Machines approach the nearer endpoint of a row, traverse it, and move to the next
row. Turn distance and turn time are optional per-transition allowances used in
bidding, execution, and centralized costs. They approximate overhead without
constructing a drivable headland path. The model does not include steering,
collision avoidance, crop damage, or machine capacity.

## Result fields

| Field | Meaning |
| --- | --- |
| `completion_time` | Simulated completion or stopping time, in seconds |
| `coverage_fraction` | Fraction of rows completed |
| `converged` | Complete row coverage |
| `allocation_stable` | Whether each allocation cycle reached a fixed point before its cap |
| `duplicate_services` | Repeated row services |
| `rows_rebid` | Accumulated changes of owner for unfinished rows after replanning |
| `reconvergence_rounds` | Post-initial allocation rounds, recorded per replan |
| `reconvergence_time` | Wall-clock allocation time per replan, in seconds |
| `msg_dropped` | Record transmissions and gossip broadcasts lost to `packet_loss` |
| `false_failure_detections` | Direct failure declarations about machines that were alive |
| `detection_delays` | Time from each failure to its first detection, in seconds |
| `planning_delay` | Accumulated simulated planning and delivery delay, in seconds |
| `planner_calls` | Centralized re-solves |

A fixed point in a disconnected component does not imply agreement across the
whole team. Initial traces count uniquely held rows whose owner records agree
across the active harvesters. Message peaks count records in one forwarding hop,
not a complete multi-hop round. Balancing counts ideal position-and-speed records;
it does not measure serialized bytes or transport time.

## Cached data

Main comparisons use ten paired seeds per condition. The heterogeneous-speed
benchmark uses forty. JSON summaries store individual values, sample means,
sample standard deviations, and 95% t intervals. Paired benchmark differences
also include standardized effects. Each file records the runtime environment
used to generate it.

An incomplete frozen plan's stopping time is not a successful makespan. Check
coverage alongside time when comparing event results.
