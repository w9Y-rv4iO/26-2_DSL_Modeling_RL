"""Study E — forming maps with RL, optimizing for a 50:50 win rate.

Studies A-D trained *players*. This one freezes the players (the heuristic
ChasePolicy and FleePolicy) and trains a policy that **edits the map**, so the
geometry is the only free variable and the win rate is attributable to it.

    balance.py    the reward signal: |runner_win_rate - 0.5| by simulation
    env.py        the edit MDP a PPO agent acts in
    baselines.py  the anchors any claim here has to beat (random / hill-climb)
    train.py      PPO over the edit MDP
    evaluate.py   re-score finalists at report sample size
    analyze.py    the tables and tests
"""

from tag_game.paths import REPO_ROOT, repo_relative  # noqa: F401  (shared helper)
