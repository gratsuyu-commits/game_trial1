"""
CrabAdventureEnv — gymnasium-compatible environment that replicates game.html physics.

Rendering (Option B):
  Create with render_mode='human'. Pygame window is created lazily on first render() call.
  train.py calls render() only on visualised episodes; non-rendered episodes skip it.
"""

import copy
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
    _PYGAME = True
except ImportError:
    _PYGAME = False

# ── Stage constants (mirrors game.html) ──────────────────────────────────────
_TILE = 32
_W, _H = 800, 400
_GH = _H - _TILE            # 368  ground Y
_GRAVITY   = 0.5
_JUMP_VY   = -11.0
_WALK_SPEED = 3.5
_MAX_VY    = 15.0
_STAGE_WIDTH = 4000
_GOAL_X = 3700
_GOAL_Y = _GH - 160         # 208

# Main ground segments — exactly mirrors game.html ground array
_GROUND: list[tuple] = [
    (0,    _GH, 1280, _TILE),   # 0   – 1280
    (1344, _GH,  896, _TILE),   # 1344 – 2240  (hole 1280-1344)
    (2336, _GH, 1504, _TILE),   # 2336 – 3840  (hole 2240-2336)
    (3968, _GH, 1792, _TILE),   # 3968 –       (hole 3840-3968)
]

_PLATFORMS: list[tuple] = [
    (400,  _GH - 96,  128, _TILE),
    (640,  _GH - 128,  96, _TILE),
    (900,  _GH - 96,  160, _TILE),
    (1600, _GH - 96,  192, _TILE),
    (1800, _GH - 160,  96, _TILE),
    (2500, _GH - 96,  160, _TILE),
    (2700, _GH - 128, 128, _TILE),
    (3200, _GH - 96,  192, _TILE),
    (3450, _GH - 160,  96, _TILE),
]

# Solids = ground + platforms (used in solid collision resolution)
_SOLIDS: list[tuple] = _GROUND + _PLATFORMS

# Blocks: (x, y) — size is always TILE×TILE
_BLOCKS: list[tuple] = [
    (300,  _GH - 96),
    (332,  _GH - 96),
    (364,  _GH - 96),
    (700,  _GH - 96),
    (1100, _GH - 128),
    (1500, _GH - 96),
    (2000, _GH - 96),
    (2600, _GH - 96),
    (3000, _GH - 96),
]

_ENEMIES_TEMPLATE: list[dict] = [
    {'x': 500.0,  'y': float(_GH-32), 'type': 'walker', 'vx': -1.2, 'vy': 0.0, 'hp': 1,  'dir': -1, 'alive': True, 'on_ground': False},
    {'x': 800.0,  'y': float(_GH-32), 'type': 'walker', 'vx': -1.2, 'vy': 0.0, 'hp': 1,  'dir': -1, 'alive': True, 'on_ground': False},
    {'x': 1000.0, 'y': float(_GH-32), 'type': 'shell',  'vx': -1.0, 'vy': 0.0, 'hp': 2,  'dir': -1, 'alive': True, 'on_ground': False},
    {'x': 1400.0, 'y': float(_GH-32), 'type': 'walker', 'vx': -1.2, 'vy': 0.0, 'hp': 1,  'dir': -1, 'alive': True, 'on_ground': False},
    {'x': 1700.0, 'y': float(_GH-32), 'type': 'spike',  'vx': -0.8, 'vy': 0.0, 'hp': 99, 'dir': -1, 'alive': True, 'on_ground': False},
    {'x': 2100.0, 'y': float(_GH-32), 'type': 'walker', 'vx': -1.2, 'vy': 0.0, 'hp': 1,  'dir': -1, 'alive': True, 'on_ground': False},
    {'x': 2400.0, 'y': float(_GH-32), 'type': 'shell',  'vx': -1.0, 'vy': 0.0, 'hp': 2,  'dir': -1, 'alive': True, 'on_ground': False},
    {'x': 2800.0, 'y': float(_GH-32), 'type': 'spike',  'vx': -0.8, 'vy': 0.0, 'hp': 99, 'dir': -1, 'alive': True, 'on_ground': False},
    {'x': 3100.0, 'y': float(_GH-32), 'type': 'walker', 'vx': -1.2, 'vy': 0.0, 'hp': 1,  'dir': -1, 'alive': True, 'on_ground': False},
    {'x': 3500.0, 'y': float(_GH-32), 'type': 'shell',  'vx': -1.0, 'vy': 0.0, 'hp': 2,  'dir': -1, 'alive': True, 'on_ground': False},
]

_ENEMY_COLORS = {
    'walker': (180, 50,  50),
    'shell':  (200, 130,  0),
    'spike':  ( 80,  0,  80),
}


def _overlap(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


class CrabAdventureEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    W, H         = _W, _H
    TILE         = _TILE
    GH           = _GH
    STAGE_WIDTH  = _STAGE_WIDTH
    GOAL_X       = _GOAL_X
    GOAL_Y       = _GOAL_Y

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(0.0, 1.0, shape=(24,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        # 0 = right | 1 = right+jump | 2 = jump | 3 = left

        # Pygame handles (created lazily in render())
        self._screen:   Any = None
        self._clock:    Any = None
        self._font:     Any = None
        self._font_q:   Any = None
        self._hud_surf: Any = None   # pre-allocated SRCALPHA surface

        # Episode state
        self._player:  dict = {}
        self._enemies: list[dict] = []
        self._agent_steps: int = 0
        self._episode:     int = 0
        self._ep_reward:  float = 0.0
        self._frame_count: int = 0

        # HUD data — train.py writes here before calling render()
        self.hud_info: dict = {}

    # ── gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._player = {
            'x': 80.0, 'y': float(_GH - 40),
            'vx': 0.0, 'vy': 0.0,
            'w': 36,   'h': 36,
            'on_ground': False,
        }
        self._enemies = copy.deepcopy(_ENEMIES_TEMPLATE)
        self._agent_steps = 0
        self._episode    += 1
        self._ep_reward   = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        x_before = self._player['x']
        terminated = truncated = False
        terminal_reward = 0.0
        cleared = False

        for _ in range(4):          # frame skip = 4
            term, tr, cl = self._step_frame(action)
            if term:
                terminated    = True
                terminal_reward = tr
                cleared       = cl
                break

        reward = terminal_reward
        if not terminated:
            dx = self._player['x'] - x_before
            if dx > 0:
                reward += 0.03 * dx   # 前進報酬を強化（0.01→0.03）
            reward -= 0.01            # survival penalty

        self._agent_steps += 1
        if not terminated and self._agent_steps >= 10_000:
            truncated = True
            reward -= 100.0

        self._ep_reward += reward
        return self._get_obs(), float(reward), terminated, truncated, {'cleared': cleared}

    def render(self):
        if not _PYGAME:
            raise RuntimeError("pip install pygame to enable rendering")
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption('Crab Adventure RL')
            self._clock    = pygame.time.Clock()
            self._font     = pygame.font.SysFont('monospace', 14)
            self._font_q   = pygame.font.SysFont('monospace', 18, bold=True)
            self._hud_surf = pygame.Surface((self.W, 34), pygame.SRCALPHA)
            self._hud_surf.fill((0, 0, 0, 115))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._draw()
        pygame.display.flip()
        self._clock.tick(60)

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    # ── Physics ───────────────────────────────────────────────────────────────

    def _step_frame(self, action: int) -> tuple[bool, float, bool]:
        """Advance 1 game frame. Returns (terminated, terminal_reward, cleared)."""
        p = self._player
        left  = (action == 3)
        right = (action in (0, 1))
        jump  = (action in (1, 2))

        if left:
            p['vx'] = -_WALK_SPEED
        elif right:
            p['vx'] = _WALK_SPEED
        else:
            p['vx'] *= 0.75

        if jump and p['on_ground']:
            p['vy'] = _JUMP_VY
            p['on_ground'] = False

        p['vy'] = min(p['vy'] + _GRAVITY, _MAX_VY)
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['x']  = max(0.0, min(p['x'], _STAGE_WIDTH - p['w']))

        p['on_ground'] = False
        self._resolve_solid(p, p['w'], p['h'])
        self._resolve_block(p)

        # Update enemies
        for e in self._enemies:
            if not e['alive']:
                continue
            e['vy'] = min(e['vy'] + _GRAVITY, _MAX_VY)
            e['x'] += e['vx']
            e['y'] += e['vy']
            e['on_ground'] = False
            self._resolve_solid(e, 24, 28)
            self._resolve_enemy_block(e)
            # Bounce at stage boundaries so enemies stay visible
            if e['x'] < 0:
                e['x'] = 0.0
                e['vx'] = abs(e['vx'])
                e['dir'] = 1
            elif e['x'] + 24 > _STAGE_WIDTH:
                e['x'] = float(_STAGE_WIDTH - 24)
                e['vx'] = -abs(e['vx'])
                e['dir'] = -1
            if e['y'] > self.H + 80:
                e['alive'] = False

        # Player-enemy interaction
        pw, ph = p['w'], p['h']
        for e in self._enemies:
            if not e['alive']:
                continue
            if not _overlap(p['x'], p['y'], pw, ph, e['x'], e['y'], 24, 28):
                continue
            stomp = p['vy'] > 0 and (p['y'] + ph) < (e['y'] + 14)
            if stomp and e['type'] != 'spike':
                e['hp'] -= 1
                p['vy'] = -7.0
                if e['hp'] <= 0:
                    e['alive'] = False
            else:
                return True, -100.0, False      # death

        # Goal
        if _overlap(p['x'], p['y'], pw, ph, _GOAL_X, _GOAL_Y, 40, 160):
            return True, 500.0, True

        # Fall into hole
        if p['y'] > self.H + 80:
            return True, -100.0, False

        return False, 0.0, False

    def _resolve_solid(self, obj: dict, ow: int, oh: int):
        """AABB collision against ground + platforms (with corner-clip fix)."""
        obj['on_ground'] = False
        for sx, sy, sw, sh in _SOLIDS:
            if not _overlap(obj['x'], obj['y'], ow, oh, sx, sy, sw, sh):
                continue
            ol  = (obj['x'] + ow) - sx
            or_ = (sx + sw)  - obj['x']
            ot  = (obj['y'] + oh) - sy
            ob  = (sy + sh)  - obj['y']
            mv  = min(ol, or_, ot, ob)
            if mv == ot:
                if obj['vy'] >= 0:          # land on top
                    obj['y'] = sy - oh
                    obj['vy'] = 0.0
                    obj['on_ground'] = True
                # vy < 0: corner clip while jumping → skip (prevents spurious sideways push)
            elif mv == ob and obj['vy'] < 0:
                obj['y'] = sy + sh
                obj['vy'] = 0.0
            elif mv == ol:
                obj['x'] = sx - ow
                obj['vx'] = 0.0
            else:
                obj['x'] = sx + sw
                obj['vx'] = 0.0

    def _resolve_block(self, p: dict):
        """AABB collision against question blocks (player only)."""
        pw, ph = p['w'], p['h']
        for bx, by in _BLOCKS:
            if not _overlap(p['x'], p['y'], pw, ph, bx, by, _TILE, _TILE):
                continue
            ol  = (p['x'] + pw) - bx
            or_ = (bx + _TILE)  - p['x']
            ot  = (p['y'] + ph) - by
            ob  = (by + _TILE)  - p['y']
            mv  = min(ol, or_, ot, ob)
            if mv == ot:
                if p['vy'] >= 0:
                    p['y'] = by - ph
                    p['vy'] = 0.0
                    p['on_ground'] = True
            elif mv == ob and p['vy'] < 0:
                p['y'] = by + _TILE
                p['vy'] = 0.0
            elif mv == ol:
                p['x'] = bx - pw
                p['vx'] = 0.0
            else:
                p['x'] = bx + _TILE
                p['vx'] = 0.0

    def _resolve_enemy_block(self, e: dict):
        """Enemy-block collision: reverse direction on horizontal hit."""
        for bx, by in _BLOCKS:
            if not _overlap(e['x'], e['y'], 24, 28, bx, by, _TILE, _TILE):
                continue
            ol  = (e['x'] + 24) - bx
            or_ = (bx + _TILE)  - e['x']
            ot  = (e['y'] + 28) - by
            ob  = (by + _TILE)  - e['y']
            mv  = min(ol, or_, ot, ob)
            if mv == ot:
                if e['vy'] >= 0:
                    e['y'] = by - 28
                    e['vy'] = 0.0
                    e['on_ground'] = True
            elif mv == ob and e['vy'] < 0:
                e['y'] = by + _TILE
                e['vy'] = 0.0
            elif mv == ol:
                e['x'] = bx - 24
                e['vx'] = -e['vx']
                e['dir'] = -e['dir']
            else:
                e['x'] = bx + _TILE
                e['vx'] = -e['vx']
                e['dir'] = -e['dir']

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        p   = self._player
        obs = np.empty(24, dtype=np.float32)

        obs[0] = float(p['x']) / _STAGE_WIDTH
        obs[1] = float(np.clip(p['y'] / self.H, 0.0, 1.0))

        # vx: [-WALK_SPEED, +WALK_SPEED] → [0, 1]
        obs[2] = float(np.clip(
            (p['vx'] + _WALK_SPEED) / (2.0 * _WALK_SPEED), 0.0, 1.0
        ))
        # vy: [JUMP_VY, MAX_VY] → [0, 1]
        obs[3] = float(np.clip(
            (p['vy'] - _JUMP_VY) / (_MAX_VY - _JUMP_VY), 0.0, 1.0
        ))

        obs[4] = 1.0 if p['on_ground'] else 0.0

        # [5..14] forward ground map (main ground only, 10 tiles)
        px = p['x']
        for i in range(10):
            obs[5 + i] = self._ground_at(px + (i + 1) * _TILE)

        # [15..22] forward enemy map (8 tiles)
        for i in range(8):
            xs = px + (i + 1) * _TILE
            obs[15 + i] = self._enemy_in(xs, xs + _TILE)

        # [23] normalised distance to goal
        obs[23] = float(max(0.0, _GOAL_X - p['x']) / _STAGE_WIDTH)

        return obs

    def _ground_at(self, x: float) -> float:
        for gx, gy, gw, gh in _GROUND:
            if gx <= x < gx + gw:
                return 1.0
        return 0.0

    def _enemy_in(self, xs: float, xe: float) -> float:
        for e in self._enemies:
            if e['alive'] and xs <= e['x'] < xe:
                return 1.0
        return 0.0

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _draw(self):
        p = self._player
        cam_x = max(0.0, min(p['x'] - self.W / 2 + p['w'] / 2,
                             _STAGE_WIDTH - self.W))
        self._frame_count += 1
        ef = (self._frame_count // 10) % 2     # animation frame (enemy + crab)

        def scr(wx: float) -> int:
            return int(wx - cam_x)

        # Sky #87CEEB
        self._screen.fill((135, 206, 235))

        # Parallax mountains — rgba(100,150,100,0.35) pre-blended over sky
        _MTN = (123, 186, 188)
        for i in range(22):
            mx = int((i * 400 - cam_x * 0.4) % (_STAGE_WIDTH + self.W))
            pygame.draw.polygon(self._screen, _MTN, [
                (mx,       _GH),
                (mx + 120, _GH - 90),
                (mx + 240, _GH),
            ])

        # Parallax clouds — rgba(255,255,255,0.85) pre-blended over sky
        _CLD = (237, 248, 252)
        _CLX = [200, 500, 900, 1300, 1700, 2100, 2500, 2900, 3300, 3700,
                4100, 4500, 4900, 5300, 5700, 6100, 6500, 6900, 7300, 7700]
        _CLY = [60, 40, 70, 50, 65, 45, 55, 70, 40, 60,
                50, 65, 45, 55, 40, 65, 50, 70, 45, 60]
        for i in range(len(_CLX)):
            cxc = int((_CLX[i] - cam_x * 0.3 + _STAGE_WIDTH + self.W)
                      % (_STAGE_WIDTH + self.W))
            cyc = _CLY[i]
            pygame.draw.circle(self._screen, _CLD, (cxc,      cyc),      18)
            pygame.draw.circle(self._screen, _CLD, (cxc + 20, cyc - 10), 23)
            pygame.draw.circle(self._screen, _CLD, (cxc + 44, cyc),      18)

        # Ground
        for gx, gy, gw, _ in _GROUND:
            rx = scr(gx)
            if rx + gw >= 0 and rx < self.W:
                self._draw_tile(rx, gy, gw)

        # Platforms
        for px_, py_, pw_, _ in _PLATFORMS:
            rx = scr(px_)
            if rx + pw_ >= 0 and rx < self.W:
                self._draw_tile(rx, py_, pw_)

        # Blocks
        for bx, by in _BLOCKS:
            rx = scr(bx)
            if rx + _TILE >= 0 and rx < self.W:
                self._draw_block(rx, by)

        # Goal
        gsx = scr(_GOAL_X)
        if -60 < gsx < self.W + 60:
            self._draw_goal(gsx, _GOAL_Y)

        # Enemies
        for e in self._enemies:
            if not e['alive']:
                continue
            esx = scr(e['x']) + 12
            esy = int(e['y']) + 14
            if -60 < esx < self.W + 60:
                if e['type'] == 'walker':
                    self._draw_walker(esx, esy, ef)
                elif e['type'] == 'shell':
                    self._draw_shell(esx, esy, e['hp'], ef)
                else:
                    self._draw_spike(esx, esy, ef)

        # Player crab
        pcx = scr(p['x']) + p['w'] // 2
        pcy = int(p['y']) + p['h'] // 2
        pdir = -1 if p.get('vx', 0) < -0.1 else 1
        self._draw_crab(pcx, pcy, pdir, ef)

        # HUD — semi-transparent bar (surface pre-allocated in render())
        self._screen.blit(self._hud_surf, (0, 0))
        hud = self.hud_info
        hud_items = [
            (f"EP:{hud.get('episode', self._episode)}",              10),
            (f"STEPS:{hud.get('total_steps', 0):,}",                200),
            (f"REWARD:{hud.get('ep_reward', self._ep_reward):.1f}", 430),
            (f"ε:{hud.get('eps', 0.0):.3f}",                  660),
        ]
        for text, x in hud_items:
            surf = self._font.render(text, True, (255, 255, 255))
            self._screen.blit(surf, (x, 10))

    def _draw_tile(self, rx: int, ry: int, rw: int):
        """Ground / platform tile: green top, brown body, bright-green tufts."""
        pygame.draw.rect(self._screen, (34, 139, 34),
                         pygame.Rect(rx, ry, rw, _TILE))
        pygame.draw.rect(self._screen, (139, 69, 19),
                         pygame.Rect(rx, ry + 8, rw, _TILE - 8))
        # Only draw tufts in the visible screen range (skip off-screen portions)
        dx0 = (max(0, -rx) // 8) * 8
        for dx in range(dx0, rw, 8):
            tx = rx + dx
            if tx >= self.W:
                break
            pygame.draw.rect(self._screen, (50, 205, 50),
                             pygame.Rect(tx, ry, 6, 5))

    def _draw_block(self, rx: int, ry: int):
        """Question block: yellow fill, brown outline, ? mark."""
        pygame.draw.rect(self._screen, (212, 160, 48),
                         pygame.Rect(rx, ry, _TILE, _TILE))
        pygame.draw.rect(self._screen, (139, 96, 16),
                         pygame.Rect(rx + 1, ry + 1, _TILE - 2, _TILE - 2), 2)
        q_surf = self._font_q.render('?', True, (240, 192, 64))
        self._screen.blit(q_surf, (rx + _TILE // 2 - q_surf.get_width() // 2,
                                   ry + _TILE - 8 - q_surf.get_height()))

    def _draw_goal(self, gsx: int, gy: int):
        """Goal: grey pole, red flag triangle, dark base."""
        pygame.draw.rect(self._screen, (136, 136, 136),
                         pygame.Rect(gsx + 10, gy - 130, 5, 162))
        pygame.draw.polygon(self._screen, (255, 68, 68), [
            (gsx + 15, gy - 130),
            (gsx + 50, gy - 112),
            (gsx + 15, gy - 94),
        ])
        pygame.draw.rect(self._screen, (102, 102, 102),
                         pygame.Rect(gsx + 2, gy + _TILE - 10, 26, 10))

    def _draw_crab(self, cx: int, cy: int, direction: int, frame: int):
        """Pixel-art crab — direct port of game.html drawCrab() fillRect calls."""
        OO = (212, 105, 74)    # #D4694A
        WW = (255, 255, 255)

        def r(x, y, w, h, c):
            rx = (cx - 22 + x) if direction >= 0 else (cx + 22 - x - w)
            pygame.draw.rect(self._screen, c,
                             pygame.Rect(rx, cy - 18 + y, w, h))

        r(4,  0,  36, 9, OO)
        r(0,  9,  44, 9, OO)
        r(8,  9,  4,  6, WW)   # left eye
        r(32, 9,  4,  6, WW)   # right eye
        r(4,  18, 36, 9, OO)
        lp = frame % 2
        d1, d2 = (2, 0) if lp else (0, 2)
        r(4,  27 + d1, 4, 9, OO)
        r(11, 27 + d2, 5, 9, OO)
        r(27, 27 + d2, 5, 9, OO)
        r(35, 27 + d1, 4, 9, OO)

    def _draw_walker(self, cx: int, cy: int, frame: int):
        """Walker enemy — port of game.html drawWalker()."""
        s = self._screen
        pygame.draw.ellipse(s, (139, 0, 0),
                            pygame.Rect(cx - 12, cy - 12, 24, 20))
        pygame.draw.circle(s, (255, 102, 102), (cx - 4, cy - 5), 3)
        pygame.draw.circle(s, (255, 102, 102), (cx + 4, cy - 5), 3)
        pygame.draw.circle(s, (255, 255, 255), (cx - 5, cy + 1), 3)
        pygame.draw.circle(s, (255, 255, 255), (cx + 5, cy + 1), 3)
        pygame.draw.circle(s, (0, 0, 0),       (cx - 4, cy + 1), 2)
        pygame.draw.circle(s, (0, 0, 0),       (cx + 6, cy + 1), 2)
        f = 2 if frame % 2 == 0 else -2
        pygame.draw.rect(s, (139, 0, 0), pygame.Rect(cx - 11, cy + 8, 8, 4 + f))
        pygame.draw.rect(s, (139, 0, 0), pygame.Rect(cx + 3,  cy + 8, 8, 4 - f))

    def _draw_shell(self, cx: int, cy: int, hp: int, frame: int):
        """Shell enemy — port of game.html drawShell()."""
        s = self._screen
        body_c = (46, 139, 87)  if hp >= 2 else (144, 238, 144)
        line_c = (26, 92, 55)   if hp >= 2 else (60, 140, 60)
        pygame.draw.ellipse(s, body_c, pygame.Rect(cx - 13, cy - 11, 26, 22))
        pygame.draw.line(s, line_c, (cx, cy - 11), (cx, cy + 11), 2)
        pygame.draw.line(s, line_c, (cx - 13, cy), (cx + 13, cy), 2)
        pygame.draw.circle(s, (255, 255, 255), (cx - 5, cy - 2), 3)
        pygame.draw.circle(s, (255, 255, 255), (cx + 5, cy - 2), 3)
        pygame.draw.circle(s, (0, 0, 0),       (cx - 4, cy - 2), 2)
        pygame.draw.circle(s, (0, 0, 0),       (cx + 6, cy - 2), 2)
        f = 1 if frame % 2 == 0 else -1
        pygame.draw.rect(s, line_c, pygame.Rect(cx - 11, cy + 10, 7, 4 + f))
        pygame.draw.rect(s, line_c, pygame.Rect(cx + 4,  cy + 10, 7, 4 - f))

    def _draw_spike(self, cx: int, cy: int, frame: int):
        """Spike enemy — port of game.html drawSpike()."""
        s = self._screen
        pygame.draw.ellipse(s, (85, 85, 85),
                            pygame.Rect(cx - 13, cy - 8, 26, 20))
        for i in range(-2, 3):
            pygame.draw.polygon(s, (136, 136, 136), [
                (cx + i * 5 - 3, cy - 7),
                (cx + i * 5,     cy - 18),
                (cx + i * 5 + 3, cy - 7),
            ])
        pygame.draw.circle(s, (255, 0, 0), (cx - 4, cy + 2), 3)
        pygame.draw.circle(s, (255, 0, 0), (cx + 4, cy + 2), 3)
        f = 1 if frame % 2 == 0 else -1
        pygame.draw.rect(s, (68, 68, 68), pygame.Rect(cx - 11, cy + 11, 7, 4 + f))
        pygame.draw.rect(s, (68, 68, 68), pygame.Rect(cx + 4,  cy + 11, 7, 4 - f))
