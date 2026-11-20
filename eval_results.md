# Sentinel detector evaluation

30 synthetic series (200 days each, seed 1), weekly pattern with 6% multiplicative noise. Each contains one spike (+40%), one drop (-40%), and one 14-day level shift (-25%).

| Mode | Window | z | Spike found | Drop found | Shift found | Shift days flagged | Mean delay (days) | False alarms / 100 days, event-free data | False alarms / 100 clean days, data with events |
|---|---|---|---|---|---|---|---|---|---|
| plain | 14 | 2.0 | 60% | 33% | 43% | 4% | 0.3 | 0.13 | 0.23 |
| plain | 14 | 2.5 | 33% | 0% | 0% | 0% | 0.0 | 0.00 | 0.02 |
| plain | 14 | 3.0 | 3% | 0% | 0% | 0% | 0.0 | 0.00 | 0.02 |
| plain | 28 | 2.0 | 57% | 33% | 60% | 5% | 0.6 | 0.04 | 0.05 |
| plain | 28 | 2.5 | 30% | 3% | 0% | 0% | 0.0 | 0.00 | 0.00 |
| plain | 28 | 3.0 | 3% | 0% | 0% | 0% | 0.0 | 0.00 | 0.00 |
| seasonal | 28 | 2.0 | 100% | 97% | 100% | 38% | 0.0 | 9.59 | 6.83 |
| seasonal | 28 | 2.5 | 100% | 97% | 100% | 25% | 0.1 | 4.63 | 3.13 |
| seasonal | 28 | 3.0 | 93% | 93% | 97% | 16% | 0.1 | 1.96 | 1.22 |
| seasonal | 56 | 2.0 | 100% | 93% | 100% | 50% | 0.1 | 7.59 | 3.59 |
| seasonal | 56 | 2.5 | 100% | 80% | 100% | 31% | 0.2 | 3.03 | 1.40 |
| seasonal | 56 | 3.0 | 93% | 73% | 100% | 19% | 0.3 | 1.06 | 0.59 |

How to read this: a detector is only useful if it finds the injected events (found columns) without crying wolf (false alarms). 'Shift days flagged' shows how much of a sustained change is still being reported once the window has absorbed it. Flags up to 3 days after an event ends are not counted as false alarms. Synthetic data is a sanity check, not proof of performance on real client data.
