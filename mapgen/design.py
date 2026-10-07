"""The map-construction MDP the PPO designer plays.

A 25x15 canvas (Lv3's frame: boundary ring with exit gates at the four
midpoints). The designer starts from an empty arena — or from a given map,
for augmentation — and places one structure per step:

    wall2   2x2 wall block          bush   2x2 bush patch (walkable, opaque)
    wallH   1x3 wall bar            item   item pad
    wallV   3x1 wall bar            warp   warp pad (consecutive pads pair up)
    gen     generator               erase  clear a 3x3 area
    STOP

Validity is enforced by an action mask, never by penalty, except for the
one rule too expensive to mask: a wall that disconnects the map is refused
at apply time (no-op + small cost).

Mask rules
  * footprints land on plain floor, never on the tiles in front of an exit
  * a wall may touch other walls but may not leave a 1-tile gap next to
    them — one-tile lanes make every chase terminal (tag_game/maps.py)
  * generators: Manhattan >= 5 from every exit and from each other
  * warp pads pair with a partner at Manhattan >= 8
  * caps: 24 bush tiles, 4 item pads, 3 warp pairs, mode's generator count

Generators the designer did not place are auto-filled (farthest-point) when
a map is scored, so every scored map is a legal round.

Reward (potential-based, surrogate-scored):
    phi(m) = -|win(m) - tau| - LAM_TO * max(0, timeout(m) - TIMEOUT_CAP)
    r_t    = phi(m_t) - phi(m_{t-1}) - STEP_COST     (+ phi(m_T) at the end)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy import ndimage

from .engine import MODES

W, H = 25, 15
FLOOR, WALL, BUSH, ITEM, WARP, GEN, EXIT = range(7)
N_CODES = 7
EXITS = ((12, 0), (0, 7), (24, 7), (12, 14))
IW, IH = W - 2, H - 2
N_CELLS = IW * IH

TYPES = ("wall2", "wallH", "wallV", "bush", "item", "warp", "gen", "erase")
FOOT = {
    "wall2": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "wallH": ((0, 0), (1, 0), (2, 0)),
    "wallV": ((0, 0), (0, 1), (0, 2)),
    "bush": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "item": ((0, 0),),
    "warp": ((0, 0),),
    "gen": ((0, 0),),
    "erase": tuple((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)),
}
N_ACTIONS = len(TYPES) * N_CELLS + 1
STOP = N_ACTIONS - 1

MAX_BUSH_TILES = 24
MAX_ITEMS = 4
MAX_WARP_PADS = 6
GEN_EXIT_DIST = 5
GEN_GAP = 5
WARP_GAP = 8
MODE_LIST = ("1v1", "2v4")

TIMEOUT_CAP = 0.20
LAM_TO = 2.0
STEP_COST = 0.002
FAIL_COST = 0.02
MIN_STEPS = 4

_YY, _XX = np.mgrid[0:H, 0:W]
_INTERIOR = np.zeros((H, W), bool)
_INTERIOR[1:-1, 1:-1] = True
_RESERVED = np.zeros((H, W), bool)
for _x, _y in EXITS:
    for _dx, _dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if 0 < _x + _dx < W - 1 and 0 < _y + _dy < H - 1:
            _RESERVED[_y + _dy, _x + _dx] = True
_EXIT_DIST = np.min([np.abs(_XX - x) + np.abs(_YY - y) for x, y in EXITS], axis=0)
_GEN_ZONE = _INTERIOR & ~_RESERVED & (_EXIT_DIST >= GEN_EXIT_DIST)
_PAD = 3


def _shift(arr, dx, dy, fill):
    """arr shifted so that out[y, x] == arr[y+dy, x+dx] for interior anchors;
    returns an (IH, IW) array."""
    p = np.pad(arr, _PAD, constant_values=fill)
    y0, x0 = _PAD + dy + 1, _PAD + dx + 1
    return p[y0:y0 + IH, x0:x0 + IW]


# (footprint offset, gap offset, wall offset) triples for the 1-tile-gap rule
_GAP_PAIRS = {}
for _t in ("wall2", "wallH", "wallV"):
    fs = set(FOOT[_t])
    pairs = []
    for ox, oy in fs:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n1, n2 = (ox + dx, oy + dy), (ox + 2 * dx, oy + 2 * dy)
            if n1 not in fs and n2 not in fs:
                pairs.append((n1, n2))
    _GAP_PAIRS[_t] = pairs


def empty_grid():
    g = np.full((H, W), FLOOR, np.int8)
    g[0, :] = g[-1, :] = WALL
    g[:, 0] = g[:, -1] = WALL
    for x, y in EXITS:
        g[y, x] = EXIT
    return g


class MapState:
    def __init__(self, grid=None, warps=None):
        self.g = empty_grid() if grid is None else grid.copy()
        self.warps = list(warps or [])  # placement order; pads 0-1, 2-3, 4-5 pair

    def copy(self):
        return MapState(self.g, self.warps)

    @classmethod
    def from_rows(cls, rows):
        code = {".": FLOOR, "#": WALL, "B": BUSH, "I": ITEM, "G": GEN, "E": EXIT,
                "T": FLOOR, "R": FLOOR, "M": GEN}
        g = empty_grid()
        pads = {}
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if (x, y) in EXITS:
                    continue
                if ch.isdigit():
                    g[y, x] = WARP
                    pads.setdefault(ch, []).append((x, y))
                else:
                    g[y, x] = code.get(ch, FLOOR)
        warps = [p for ch in sorted(pads) for p in pads[ch]]
        return cls(g, warps)

    def n_gens(self):
        return int((self.g == GEN).sum())

    def export_grid(self):
        """Grid as the game sees it: an unpaired last warp pad is plain floor."""
        g = self.g.copy()
        if len(self.warps) % 2 == 1:
            x, y = self.warps[-1]
            g[y, x] = FLOOR
        return g

    def filled(self, n_gen):
        """Copy with generators auto-placed (farthest-point) up to n_gen."""
        s = self.copy()
        gens = [(int(x), int(y)) for y, x in zip(*np.nonzero(s.g == GEN))]
        while len(gens) < n_gen:
            ok = _GEN_ZONE & (s.g == FLOOR)
            if gens:
                d = np.min([np.abs(_XX - x) + np.abs(_YY - y) for x, y in gens], axis=0)
            else:
                d = _EXIT_DIST.copy()
            score = np.where(ok, d, -1)
            if score.max() < 0:
                break
            y, x = np.unravel_index(int(np.argmax(score)), score.shape)
            s.g[y, x] = GEN
            gens.append((int(x), int(y)))
        return s

    def to_rows(self):
        g = self.export_grid()
        chars = {FLOOR: ".", WALL: "#", BUSH: "B", ITEM: "I", GEN: "G", EXIT: "E",
                 WARP: "."}
        out = [[chars[int(c)] for c in row] for row in g]
        for i in range(len(self.warps) // 2):
            for x, y in self.warps[2 * i:2 * i + 2]:
                out[y][x] = str(i + 1)
        return ["".join(r) for r in out]

    # ------------------------------------------------------------ actions
    def action_mask(self, n_gen, t=MIN_STEPS):
        g = self.g
        free = (g == FLOOR) & ~_RESERVED
        wallish = g == WALL
        walk = ~wallish
        mask = np.zeros(N_ACTIONS, bool)

        def put(i, arr):
            mask[i * N_CELLS:(i + 1) * N_CELLS] = arr.ravel()

        for i, typ in enumerate(TYPES):
            if typ == "erase":
                removable = _INTERIOR & np.isin(g, (WALL, BUSH, ITEM, WARP, GEN))
                ok = np.zeros((IH, IW), bool)
                for dx, dy in FOOT[typ]:
                    ok |= _shift(removable, dx, dy, False)
                put(i, ok)
                continue
            ok = np.ones((IH, IW), bool)
            for dx, dy in FOOT[typ]:
                ok &= _shift(free, dx, dy, False)
            if typ in _GAP_PAIRS:
                bad = np.zeros((IH, IW), bool)
                for (x1, y1), (x2, y2) in _GAP_PAIRS[typ]:
                    bad |= _shift(walk, x1, y1, False) & _shift(wallish, x2, y2, True)
                ok &= ~bad
            elif typ == "bush":
                if (g == BUSH).sum() + 4 > MAX_BUSH_TILES:
                    ok[:] = False
            elif typ == "item":
                if (g == ITEM).sum() >= MAX_ITEMS:
                    ok[:] = False
            elif typ == "warp":
                if len(self.warps) >= MAX_WARP_PADS:
                    ok[:] = False
                elif len(self.warps) % 2 == 1:
                    px, py = self.warps[-1]
                    ok &= (np.abs(_XX - px) + np.abs(_YY - py))[1:-1, 1:-1] >= WARP_GAP
            elif typ == "gen":
                ok &= _GEN_ZONE[1:-1, 1:-1]
                gens = list(zip(*np.nonzero(g == GEN)))
                if len(gens) >= n_gen:
                    ok[:] = False
                elif gens:
                    d = np.min([np.abs(_XX - x) + np.abs(_YY - y) for y, x in gens], axis=0)
                    ok &= d[1:-1, 1:-1] >= GEN_GAP
            put(i, ok)
        mask[STOP] = t >= MIN_STEPS
        return mask

    def apply(self, action):
        """Returns True if the map changed, False if the edit was refused."""
        if action == STOP:
            return False
        ti, cell = divmod(int(action), N_CELLS)
        typ = TYPES[ti]
        x, y = cell % IW + 1, cell // IW + 1
        cells = [(x + dx, y + dy) for dx, dy in FOOT[typ]
                 if 0 < x + dx < W - 1 and 0 < y + dy < H - 1]
        if typ == "erase":
            for cx, cy in cells:
                if self.g[cy, cx] == WARP:
                    self._remove_warp((cx, cy))
                elif self.g[cy, cx] in (WALL, BUSH, ITEM, GEN):
                    self.g[cy, cx] = FLOOR
            return True
        code = {"wall2": WALL, "wallH": WALL, "wallV": WALL, "bush": BUSH,
                "item": ITEM, "warp": WARP, "gen": GEN}[typ]
        old = [self.g[cy, cx] for cx, cy in cells]
        for cx, cy in cells:
            self.g[cy, cx] = code
        if code == WALL and not connected(self.g):
            for (cx, cy), o in zip(cells, old):
                self.g[cy, cx] = o
            return False
        if code == WARP:
            self.warps.append((x, y))
        return True

    def _remove_warp(self, pos):
        i = self.warps.index(pos)
        j = i + 1 if i % 2 == 0 else i - 1
        drop = {i} | ({j} if j < len(self.warps) else set())
        for k in sorted(drop, reverse=True):
            px, py = self.warps.pop(k)
            self.g[py, px] = FLOOR


def connected(g):
    _, n = ndimage.label(g != WALL)
    return n == 1


def encode_map(state, mode):
    """(8, H, W) float32: one-hot tile codes of the game-visible grid + mode."""
    g = state.export_grid()
    x = np.zeros((N_CODES + 1, H, W), np.float32)
    for c in range(N_CODES):
        x[c] = g == c
    x[N_CODES] = 1.0 if mode == "2v4" else 0.0
    return x


class DesignEnv(gym.Env):
    """One episode = one map. `scorer.predict(states, modes)` returns an
    (n, 2) array of predicted [runner win rate, timeout rate]."""

    metadata = {"render_modes": []}
    N_OBS_CH = N_CODES + 5  # tiles, tau, win_hat, timeout_hat, progress, mode

    def __init__(self, scorer, T=24, tau_range=(0.25, 0.75), modes=MODE_LIST,
                 starts=None, start_p=0.0, fixed_tau=None, fixed_mode=None):
        super().__init__()
        self.scorer = scorer
        self.T = T
        self.tau_range = tau_range
        self.modes = modes
        self.starts = starts or []
        self.start_p = start_p
        self.fixed_tau = fixed_tau
        self.fixed_mode = fixed_mode
        self.observation_space = spaces.Box(0.0, 1.0, (self.N_OBS_CH, H, W), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

    # --------------------------------------------------------------------
    # Reset and step are split into a state change (begin_*) and a scoring
    # half (end_*) so a vectorized wrapper can score every env's map in ONE
    # surrogate batch. reset()/step() below chain the halves for single use.
    def scoring_state(self):
        return self.st.filled(MODES[self.mode]["n_generators"])

    def _phi(self, pred):
        self.pred = (float(pred[0]), float(pred[1]))
        return (-abs(self.pred[0] - self.tau)
                - LAM_TO * max(0.0, self.pred[1] - TIMEOUT_CAP))

    def begin_reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        self.mode = options.get("mode") or self.fixed_mode or self.modes[
            int(rng.integers(len(self.modes)))]
        tau = options.get("tau", self.fixed_tau)
        self.tau = float(tau if tau is not None else rng.uniform(*self.tau_range))
        if self.starts and rng.random() < self.start_p:
            self.st = self.starts[int(rng.integers(len(self.starts)))].copy()
        else:
            self.st = MapState()
        self.t = 0
        self.fails = 0

    def end_reset(self, pred):
        self.phi = self._phi(pred)
        return self._obs()

    def begin_step(self, action):
        done = action == STOP
        reward = -STEP_COST
        if not done and not self.st.apply(action):
            reward -= FAIL_COST
            self.fails += 1
        self.t += 1
        self._pending = (done or self.t >= self.T, reward)

    def end_step(self, pred):
        done, reward = self._pending
        phi = self._phi(pred)
        reward += phi - self.phi
        self.phi = phi
        info = {}
        if done:
            reward += phi
            info = {"tau": self.tau, "mode": self.mode, "win_hat": self.pred[0],
                    "timeout_hat": self.pred[1], "steps": self.t}
        return self._obs(), float(reward), bool(done), False, info

    def _predict_self(self):
        return self.scorer.predict([self.scoring_state()], [self.mode])[0]

    def reset(self, seed=None, options=None):
        self.begin_reset(seed, options)
        return self.end_reset(self._predict_self()), {}

    def action_masks(self):
        return self.st.action_mask(MODES[self.mode]["n_generators"], self.t)

    def step(self, action):
        self.begin_step(action)
        return self.end_step(self._predict_self())

    def _obs(self):
        x = np.zeros((self.N_OBS_CH, H, W), np.float32)
        x[:N_CODES] = encode_map(self.st, self.mode)[:N_CODES]
        x[N_CODES] = self.tau
        x[N_CODES + 1] = self.pred[0]
        x[N_CODES + 2] = self.pred[1]
        x[N_CODES + 3] = self.t / self.T
        x[N_CODES + 4] = 1.0 if self.mode == "2v4" else 0.0
        return x
