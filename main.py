import math
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT
)

CENTER_X = CENTER_Y = CENTER
SUN_R = 10.0
MAX_SPEED = 6.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
LAUNCH_CLEARANCE = 0.1
INTERCEPT_TOLERANCE = 1
HORIZON = 110
ROUTE_SEARCH_HORIZON = 60


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def is_static_planet(planet, initial_by_id):
    init = initial_by_id.get(planet.id, planet)
    return dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius >= ROTATION_LIMIT


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = max(0.0, min(1.0, math.log(ships) / math.log(1000.0)))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    return dist(px, py, x1 + t * dx, y1 + t * dy)


def segment_hits_sun(x1, y1, x2, y2, safety=SUN_SAFETY):
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety


def launch_point(sx, sy, sr, angle):
    c = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * c, sy + math.sin(angle) * c


def actual_path_geometry(sx, sy, sr, tx, ty, tr):
    angle = math.atan2(ty - sy, tx - sx)
    lx, ly = launch_point(sx, sy, sr, angle)
    hit_dist = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    return angle, lx, ly, lx + math.cos(angle) * hit_dist, ly + math.sin(angle) * hit_dist, hit_dist


def safe_angle_and_distance(sx, sy, sr, tx, ty, tr):
    angle, lx, ly, ex, ey, hit_dist = actual_path_geometry(sx, sy, sr, tx, ty, tr)
    if segment_hits_sun(lx, ly, ex, ey):
        return None
    return angle, hit_dist


def predict_planet_position(planet, initial_by_id, angular_velocity, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return CENTER_X + r * math.cos(new_ang), CENTER_Y + r * math.sin(new_ang)


def estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def travel_time(sx, sy, sr, tx, ty, tr, ships):
    est = estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    return est[1] if est else 10 ** 9


def search_safe_intercept(src, target, ships, initial_by_id, ang_vel):
    best, best_score = None, None
    for candidate_turns in range(1, min(HORIZON, ROUTE_SEARCH_HORIZON) + 1):
        pos = predict_planet_position(target, initial_by_id, ang_vel, candidate_turns)
        est = estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
        if est is None:
            continue
        _, turns = est
        if abs(turns - candidate_turns) > INTERCEPT_TOLERANCE:
            continue
        actual_turns = max(turns, candidate_turns)
        actual_pos = predict_planet_position(target, initial_by_id, ang_vel, actual_turns)
        confirm = estimate_arrival(src.x, src.y, src.radius, actual_pos[0], actual_pos[1], target.radius, ships)
        if confirm is None:
            continue
        delta = abs(confirm[1] - actual_turns)
        if delta > INTERCEPT_TOLERANCE:
            continue
        score = (delta, confirm[1], candidate_turns)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], actual_pos[0], actual_pos[1])
    return best


def aim_with_prediction(src, target, ships, initial_by_id, ang_vel):
    est = estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    target_is_static = is_static_planet(target, initial_by_id)

    if est is None:
        return None if target_is_static else search_safe_intercept(src, target, ships, initial_by_id, ang_vel)

    tx, ty = target.x, target.y
    for _ in range(5):
        _, turns = est
        ntx, nty = predict_planet_position(target, initial_by_id, ang_vel, turns)
        next_est = estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if next_est is None:
            return None if target_is_static else search_safe_intercept(src, target, ships, initial_by_id, ang_vel)
        if abs(ntx - tx) < 0.3 and abs(nty - ty) < 0.3 and abs(next_est[1] - turns) <= INTERCEPT_TOLERANCE:
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est

    final_est = estimate_arrival(src.x, src.y, src.radius, tx, ty, target.radius, ships)
    if final_est is None:
        return search_safe_intercept(src, target, ships, initial_by_id, ang_vel)
    return final_est[0], final_est[1], tx, ty


class TacticalQuadrantAgent:
    """Implements defensive tanks, reinforcements, and peacetime buffing."""
    def __init__(self):
        self._initial_by_id = {}
        self._targeted = {}
        self.current_turn = 0
        self.last_turn_ships = {}
        self.last_turn_owner = {}
        self.last_turn_ejected = {}

    def __call__(self, obs, config=None):
        self.current_turn += 1
        for tid, expire_turn in list(self._targeted.items()):
            if self.current_turn >= expire_turn:
                del self._targeted[tid]

        player = obs.player if hasattr(obs, "player") else obs.get("player", 0)
        raw_p  = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
        ang_vel = obs.angular_velocity if hasattr(obs, "angular_velocity") else obs.get("angular_velocity", 0.03)
        comet_ids = set(obs.comet_planet_ids if hasattr(obs, "comet_planet_ids") else obs.get("comet_planet_ids", []))

        all_planets = [Planet(*p) for p in raw_p]
        planets = [p for p in all_planets if p.id not in comet_ids]

        if not self._initial_by_id:
            self._initial_by_id = {p.id: p for p in all_planets}

        under_attack_ids = set()
        reinforcement_needs = {}
        for p in planets:
            if p.id in getattr(self, 'last_turn_owner', {}) and self.last_turn_owner[p.id] == player and p.owner == player:
                ejected = self.last_turn_ejected.get(p.id, 0)
                expected = self.last_turn_ships[p.id] + p.production - ejected
                if p.ships < expected:
                    lost = expected - p.ships
                    under_attack_ids.add(p.id)
                    reinforcement_needs[p.id] = lost + 1

        my_pl = [p for p in planets if p.owner == player]
        if not my_pl: 
            self.last_turn_ships = {p.id: p.ships for p in planets}
            self.last_turn_owner = {p.id: p.owner for p in planets}
            self.last_turn_ejected = {}
            return []

        def get_quadrant(p):
            cx, cy = CENTER_X, CENTER_Y
            x, y = p.x - cx, p.y - cy
            if x >= 0 and y >= 0: return 0
            if x < 0 and y >= 0: return 1
            if x < 0 and y < 0: return 2
            return 3

        def get_prev_quadrant(q):
            if ang_vel >= 0: return (q - 1) % 4
            else: return (q + 1) % 4

        my_static = [p for p in my_pl if is_static_planet(p, self._initial_by_id)]
        my_moving = [p for p in my_pl if not is_static_planet(p, self._initial_by_id)]

        unoccupied = [p for p in planets if p.owner != player and p.id not in self._targeted]
        unocc_static = [p for p in unoccupied if is_static_planet(p, self._initial_by_id)]
        unocc_moving = [p for p in unoccupied if not is_static_planet(p, self._initial_by_id)]

        moves = []
        available_ships = {p.id: p.ships for p in my_pl}

        def aim_and_send(src, tgt, needed_ships):
            best_aim = None
            needed = needed_ships
            for _ in range(4):
                best_aim = aim_with_prediction(src, tgt, needed, self._initial_by_id, ang_vel)
                if best_aim is None: break
                angle, turns, tx, ty = best_aim
                if tgt.owner == player:
                    new_needed = needed_ships
                elif tgt.owner == -1: 
                    new_needed = tgt.ships + 1
                else: 
                    new_needed = int(tgt.ships + tgt.production * turns) + 1
                if new_needed == needed: break
                needed = new_needed

            if best_aim and available_ships[src.id] > needed:
                moves.append([src.id, best_aim[0], needed])
                available_ships[src.id] -= needed
                if tgt.owner != player:
                    self._targeted[tgt.id] = self.current_turn + turns + 2
                return True
            return False

        for mp in my_moving:
            if not my_static:
                if unocc_static:
                    tgt = min(unocc_static, key=lambda t: (t.ships, dist(mp.x, mp.y, t.x, t.y)))
                    aim_and_send(mp, tgt, tgt.ships + 1)
            else:
                pass

        my_static_by_quad = {i: [] for i in range(4)}
        for p in my_static:
            my_static_by_quad[get_quadrant(p)].append(p)

        all_static = [p for p in planets if is_static_planet(p, self._initial_by_id)]
        quadrant_static_owned = {}
        for q in range(4):
            static_in_q = [p for p in all_static if get_quadrant(p) == q]
            if not static_in_q: quadrant_static_owned[q] = True
            else: quadrant_static_owned[q] = all(p.owner == player for p in static_in_q)

        for q in range(4):
            if not my_static_by_quad[q]: continue

            my_q_static = my_static_by_quad[q]
            tanks = [p for p in my_q_static if p.production >= 3]
            small_peeps = [p for p in my_q_static if p.production < 3]

            attacked_in_q = [p for p in my_pl if p.id in under_attack_ids and get_quadrant(p) == q]
            enemies_in_q = [p for p in unoccupied if get_quadrant(p) == q]
            heavy_enemies = [p for p in enemies_in_q if p.production >= 3]
            small_enemies = [p for p in enemies_in_q if p.production < 3]

            # --- Phase 1: Reinforcements ---
            for target_p in attacked_in_q:
                needed = reinforcement_needs[target_p.id]
                helpers = [p for p in my_q_static if p.id != target_p.id]
                if not helpers: continue
                send_amount = math.ceil(needed / len(helpers))
                for helper in helpers:
                    aim_and_send(helper, target_p, send_amount)

            # --- Phase 2: Defense / Expansion in Quadrant ---
            if enemies_in_q:
                for tank in tanks:
                    tgt = None
                    valid_heavy = [t for t in heavy_enemies if t.ships < available_ships[tank.id]]
                    if valid_heavy: tgt = min(valid_heavy, key=lambda t: dist(tank.x, tank.y, t.x, t.y))
                    if not tgt:
                        valid_small = [t for t in small_enemies if t.ships < available_ships[tank.id]]
                        if valid_small: tgt = min(valid_small, key=lambda t: dist(tank.x, tank.y, t.x, t.y))
                    if tgt: aim_and_send(tank, tgt, tgt.ships + 1)

                for peep in small_peeps:
                    tgt = None
                    valid_small = [t for t in small_enemies if t.ships < available_ships[peep.id]]
                    if valid_small: tgt = min(valid_small, key=lambda t: dist(peep.x, peep.y, t.x, t.y))
                    if not tgt:
                        valid_heavy = [t for t in heavy_enemies if t.ships < available_ships[peep.id]]
                        if valid_heavy: tgt = min(valid_heavy, key=lambda t: dist(peep.x, peep.y, t.x, t.y))
                    if tgt: aim_and_send(peep, tgt, tgt.ships + 1)

            # --- Phase 3: Prev Quadrant Enemies (Shared Fleets) ---
            elif quadrant_static_owned[q]:
                prev_q = get_prev_quadrant(q)
                coming_planets = [p for p in unocc_moving if get_quadrant(p) == prev_q]
                if coming_planets:
                    tgt = min(coming_planets, key=lambda cp: min(dist(sp.x, sp.y, cp.x, cp.y) for sp in my_q_static))
                    sp0 = my_q_static[0]
                    best_aim = None
                    needed = tgt.ships + 1
                    for _ in range(4):
                        best_aim = aim_with_prediction(sp0, tgt, needed, self._initial_by_id, ang_vel)
                        if best_aim is None: break
                        _, turns, _, _ = best_aim
                        if tgt.owner == -1: new_needed = tgt.ships + 1
                        else: new_needed = int(tgt.ships + tgt.production * turns) + 1
                        if new_needed == needed: break
                        needed = new_needed

                    if best_aim:
                        send_amount = math.ceil(needed / len(my_q_static))
                        for sp in my_q_static:
                            aim_and_send(sp, tgt, send_amount)

            # --- Phase 4: Peacetime Buff ---
            if not enemies_in_q:
                prev_q = get_prev_quadrant(q)
                enemies_in_prev = [p for p in unoccupied if get_quadrant(p) == prev_q]
                if not enemies_in_prev:
                    for tank in tanks:
                        weak_peeps = [p for p in small_peeps if p.ships < 30 and available_ships[p.id] < 30]
                        if weak_peeps:
                            tgt = min(weak_peeps, key=lambda p: p.ships)
                            send_amount = max(1, int(tank.production * 1))
                            aim_and_send(tank, tgt, send_amount)

        # Moving planets normal logic
        for mp in my_moving:
            if my_static:
                q = get_quadrant(mp)
                if quadrant_static_owned[q]:
                    if unocc_moving:
                        tgt = min(unocc_moving, key=lambda t: dist(mp.x, mp.y, t.x, t.y))
                        aim_and_send(mp, tgt, tgt.ships + 1)
                else:
                    unocc_static_in_q = [p for p in unocc_static if get_quadrant(p) == q]
                    if unocc_static_in_q:
                        tgt = min(unocc_static_in_q, key=lambda t: dist(mp.x, mp.y, t.x, t.y))
                        aim_and_send(mp, tgt, tgt.ships + 1)

        self.last_turn_ships = {p.id: p.ships for p in planets}
        self.last_turn_owner = {p.id: p.owner for p in planets}
        self.last_turn_ejected = {}
        for move in moves:
            src_id, angle, ships = move
            self.last_turn_ejected[src_id] = self.last_turn_ejected.get(src_id, 0) + ships

        return moves

agent_instance = TacticalQuadrantAgent()
def agent(obs, config=None):
    return agent_instance(obs, config)
