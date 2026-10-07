"""Fast checks for the map-designer study (mapgen/).

    python tests/test_mapgen.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from mapgen.design import (N_CELLS, STOP, TYPES, WALL, MapState,  # noqa: E402
                           connected)
from mapgen.engine import MODES, TeamController, TeamWorld, play  # noqa: E402
from tag_game.world import TILE  # noqa: E402

ROOM = [
    "#########",
    "#1.....1#",
    "#.......#",
    "#..B....#",
    "#...G...#",
    "#.......#",
    "E.......#",
    "#########",
]


def test_warp_in_one_room():
    w = TeamWorld({"Main": ROOM}, seed=0)
    r = w.runner
    r.x, r.y = (1.5 * TILE, 1.5 * TILE)
    r._door_ready = True
    w._use_doors(r)
    assert (r.room, r.tile) == ("Main", (7, 1)), r.tile


def test_bush_blocks_sight():
    w = TeamWorld({"Main": ROOM}, seed=0)
    c = lambda tx, ty: ((tx + 0.5) * TILE, (ty + 0.5) * TILE)  # noqa: E731
    assert not w.line_of_sight("Main", *c(1, 3), *c(5, 3))   # through the bush at (3, 3)
    assert w.line_of_sight("Main", *c(1, 2), *c(5, 2))       # the row above is clear
    assert w.line_of_sight("Main", *c(3, 3), *c(6, 3))       # looking out of a bush is fine
    assert not w.line_of_sight("Main", *c(6, 3), *c(3, 3))   # looking into one is not


def test_team_outcome():
    w = TeamWorld({"Main": ROOM}, n_taggers=2, n_runners=4, required_generators=1, seed=3)
    assert w.need == 2 and len(w.taggers) == 2 and len(w.runners) == 4
    for r in w.runners[:3]:
        r.status = "caught"
    w.step([(0.0, 0.0)] * 6, 1 / 15)
    assert w.winner == "tagger" and w.end_reason == "tag"


def test_heuristics_finish_rounds():
    from tag_game.maps import MAPS
    for mode in MODES:
        out = play(MAPS["Lv3"]["Main"], mode, seed=1)
        assert out["time"] <= 90.0 + 1e-6 and out["win"] in (0, 1), out


def test_designer_maps_are_legal():
    rng = np.random.default_rng(0)
    for i in range(30):
        mode = ("1v1", "2v4")[i % 2]
        n_gen = MODES[mode]["n_generators"]
        st = MapState()
        for _ in range(25):
            mask = st.action_mask(n_gen)
            st.apply(int(rng.choice(np.flatnonzero(mask[:STOP]))))
            assert connected(st.g)
        f = st.filled(n_gen)
        assert f.n_gens() == n_gen
        rows = f.to_rows()
        assert (MapState.from_rows(rows).export_grid() == f.export_grid()).all()
        w = TeamWorld({"Main": rows}, seed=i)
        TeamController(w).act(w)
    assert len(TYPES) * N_CELLS + 1 == STOP + 1


def test_gap_rule():
    st = MapState()
    n_gen = MODES["1v1"]["n_generators"]
    st.apply(0 * N_CELLS + (5 - 1) * 23 + (5 - 1))  # wall2 at (5, 5)
    mask = st.action_mask(n_gen)
    # a 2x2 block at (8, 5) would leave a 1-tile lane at x = 7
    assert not mask[0 * N_CELLS + (5 - 1) * 23 + (8 - 1)]
    # touching (x = 7) or a 2-tile gap (x = 9) is allowed
    assert mask[0 * N_CELLS + (5 - 1) * 23 + (7 - 1)]
    assert mask[0 * N_CELLS + (5 - 1) * 23 + (9 - 1)]
    assert (st.g == WALL).sum() > 0


def test_neighbour_cache_follows_wall_changes():
    """`mapgen.engine` patches `agents._neighbors` for the whole process, so
    the cache has to stay correct for rounds mapgen never runs. `journey`
    carves wall tiles open mid-round; an unstamped cache kept answering with
    the pre-carve geometry, which silently re-routes every heuristic in any
    process that happened to import mapgen.
    """
    from tag_game import agents as _A
    from tag_game.skills import JOURNEY
    from tag_game.world import RUNNER, World

    spec = {"Main": ["#############",
                     "#R...#.....T#",
                     "#....#......#",
                     "#....#......#",
                     "#....#......#",
                     "#..........M#",
                     "#############"]}
    w = World(spec, skills_enabled=True, items_enabled=False,
              round_time=600.0, seed=1)
    runner = w.agents[RUNNER]
    runner.room = "Main"
    runner.x, runner.y = 2.5 * TILE, 3.5 * TILE
    runner.facing = (0.0, -1.0)          # up, into the wall row at y = 0

    here, above = ("Main", (2, 3)), ("Main", (2, 2))
    assert above in set(_A._neighbors(w, here))   # open floor, cached now

    w.place_wall(runner, (2, 2), 6.0)             # character_3's Q closes it
    assert above not in set(_A._neighbors(w, here)), "stale cache after wall"

    for _ in range(100):                          # let the placement expire
        w.step({}, 1 / 15)
        if not w.closed_tiles("Main"):
            break
    assert above in set(_A._neighbors(w, here)), "stale cache after expiry"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
