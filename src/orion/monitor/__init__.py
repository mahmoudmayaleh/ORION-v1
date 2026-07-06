"""Population-level strategy monitoring (v6.2 §6.5, Phase 4 §4.8).

Page-Hinkley change detectors over slice resolution streams. When persistent
rejection patterns are detected (either per-plan or globally), the trainer
asynchronously asks Agent B to update its planning style — replanning is a
*population-level* event, never a per-slice MDO action.

Modules:
    page_hinkley       PH change detector — online, O(1) memory per stream.
    strategy_monitor   per-plan and global PH streams + refresh triggers.
"""
