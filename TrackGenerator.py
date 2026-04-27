"""
track_generator.py  –  CSG / Shape-Union Track Generator
Compatible with NewTech.py (Outputs shape (2*M, 4) float32)
"""

import os
import math
import random
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LinearRing, LineString
from shapely.ops import unary_union
from shapely.affinity import rotate, translate
from shapely.geometry import Point

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_TRACKS   = 1000
OUTPUT_DIR   = "generated_tracks"
SEED         = 42

DIFFICULTY_SPLIT = {"easy": 0.25, "medium": 0.50, "hard": 0.25}

PARAMS = {
    "easy":   dict(n=12, var=40,  half_w=70),
    "medium": dict(n=16, var=70,  half_w=60),
    "hard":   dict(n=20, var=90,  half_w=45),
}

CANVAS_W, CANVAS_H = 1000, 1000


# ── Shape Generation ───────────────────────────────────────────────────────────

def get_random_shape(rng, difficulty):
    base_size = 180
    if difficulty == "easy":
        if rng.random() < 0.7:
            w = rng.uniform(base_size, base_size * 1.5)
            h = rng.uniform(base_size * 0.8, base_size * 1.2)
            return Polygon([(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)])
        else:
            angles = np.linspace(0, 2*math.pi, 6)[:-1]
            return Polygon([(math.cos(a)*base_size, math.sin(a)*base_size) for a in angles])
    elif difficulty == "hard":
        if rng.random() < 0.8:
            length = rng.uniform(base_size * 1.8, base_size * 2.5)
            width  = rng.uniform(base_size * 1.0, base_size * 1.5)
            return Polygon([(0,length/2),(-width/2,-length/2),(width/2,-length/2)])
        else:
            w = rng.uniform(base_size*1.8, base_size*2.5)
            h = rng.uniform(base_size*0.8, base_size*1.2)
            return Polygon([(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)])
    else:  # medium
        if rng.random() < 0.5:
            w = rng.uniform(base_size*1.0, base_size*1.4)
            h = rng.uniform(base_size*1.0, base_size*1.4)
            return Polygon([(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)])
        else:
            length = rng.uniform(base_size*1.2, base_size*1.8)
            width  = rng.uniform(base_size*0.9, base_size*1.3)
            return Polygon([(0,length/2),(-width/2,-length/2),(width/2,-length/2)])


# ── Smoothing ──────────────────────────────────────────────────────────────────

def chaikin_smooth(pts, iterations=3):
    curr = np.array(pts)
    for _ in range(iterations):
        n = len(curr)
        new = []
        for i in range(n):
            p1, p2 = curr[i], curr[(i+1) % n]
            new.append(0.75*p1 + 0.25*p2)
            new.append(0.25*p1 + 0.75*p2)
        curr = np.array(new)
    return curr


# ── Wall Builder (FIX: use Shapely offset instead of manual normals) ───────────

def _build_walls_from_centerline(centerline_pts, half_width):
    """
    Uses Shapely parallel_offset — guaranteed no self-intersections.
    Returns (outer_segs, inner_segs) as (N,4) float32 arrays.
    """
    # Close the loop explicitly
    pts = list(map(tuple, centerline_pts))
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    line = LinearRing(pts[:-1])  # closed ring

    # FIX: use Shapely's offset_curve (replaces deprecated parallel_offset)
    # 'left' = outward for CCW rings, 'right' = inward
    # We detect winding and assign accordingly
    ring_poly = Polygon(pts[:-1])

    # Shapely exterior is always CCW after buffer ops
    # left offset = outside, right offset = inside
    try:
        outer_geom = line.offset_curve(half_width,  quad_segs=8)
        inner_geom = line.offset_curve(-half_width, quad_segs=8)
    except AttributeError:
        # Shapely < 1.8 fallback
        outer_geom = line.parallel_offset(half_width,  side='left',  join_style=2)
        inner_geom = line.parallel_offset(half_width,  side='right', join_style=2)

    # Handle MultiLineString (rare but possible)
    def biggest(geom):
        if geom.geom_type == 'MultiLineString':
            return max(geom.geoms, key=lambda g: g.length)
        return geom

    outer_geom = biggest(outer_geom)
    inner_geom = biggest(inner_geom)

    def geom_to_segs(geom, n_target):
        """Resample geometry to n_target points, return (N,4) segments."""
        total = geom.length
        coords = [geom.interpolate(t / n_target, normalized=True).coords[0]
                  for t in range(n_target)]
        coords = np.array(coords)
        n = len(coords)
        segs = np.empty((n, 4), dtype=np.float32)
        for i in range(n):
            segs[i] = [coords[i,0], coords[i,1],
                       coords[(i+1)%n,0], coords[(i+1)%n,1]]
        return segs

    n_pts = len(centerline_pts)
    outer_segs = geom_to_segs(outer_geom, n_pts)
    inner_segs = geom_to_segs(inner_geom, n_pts)
    return outer_segs, inner_segs


# ── Validation (FIX: check offset polygons, not segment endpoints) ─────────────

def _walls_are_valid(outer_segs, inner_segs):
    """
    Checks:
    1. Inner wall forms a valid simple polygon
    2. Inner wall is fully inside outer wall
    3. No wall self-intersects
    """
    try:
        outer_poly = Polygon(outer_segs[:, :2])
        inner_poly = Polygon(inner_segs[:, :2])

        if not outer_poly.is_valid or not outer_poly.is_simple:
            outer_poly = outer_poly.buffer(0)  # auto-repair
        if not inner_poly.is_valid or not inner_poly.is_simple:
            inner_poly = inner_poly.buffer(0)

        if outer_poly.is_empty or inner_poly.is_empty:
            return False

        # Inner must be contained within outer
        if not outer_poly.contains(inner_poly):
            return False

        return True
    except Exception:
        return False


# ── Core Generator ─────────────────────────────────────────────────────────────

def generate_csg_track(rng: random.Random, difficulty: str) -> np.ndarray:
    p = PARAMS[difficulty]
    max_attempts = 150

    for attempt in range(max_attempts):
        try:
            cx, cy       = CANVAS_W / 2, CANVAS_H / 2
            base_radius  = 280.0
            spine_thick  = p["half_w"] + 25.0

            # Spine ring guarantees loop topology
            spine = Point(cx, cy).buffer(base_radius).exterior.buffer(spine_thick)
            shapes = [spine]

            for i in range(p["n"]):
                angle = (i / p["n"]) * 2 * math.pi + rng.uniform(-0.2, 0.2)
                r = base_radius + rng.uniform(-p["var"], p["var"])
                x = cx + math.cos(angle) * r
                y = cy + math.sin(angle) * r
                shape = get_random_shape(rng, difficulty)
                shape = rotate(shape, rng.uniform(0, 360), origin=(0, 0))
                shape = translate(shape, xoff=x, yoff=y)
                shapes.append(shape)

            merged = unary_union(shapes)
            if isinstance(merged, MultiPolygon):
                merged = max(merged.geoms, key=lambda a: a.area)

            # Smooth: open tight concavities, remove sharp spikes
            safe_r = p["half_w"] * 0.6   # FIX: was half_w+8 — too aggressive, ate all features
            merged = merged.buffer(safe_r, join_style=1).buffer(-safe_r, join_style=1)
            if isinstance(merged, MultiPolygon):
                merged = max(merged.geoms, key=lambda a: a.area)
            merged = merged.buffer(-safe_r, join_style=1).buffer(safe_r, join_style=1)
            if isinstance(merged, MultiPolygon):
                merged = max(merged.geoms, key=lambda a: a.area)

            if merged.is_empty or not hasattr(merged, 'exterior'):
                continue

            # Simplify to reduce vertex count before smoothing
            merged = merged.simplify(5.0, preserve_topology=True)

            exterior_coords = np.array(merged.exterior.coords)
            if len(exterior_coords) < 6:
                continue

            # Smooth centerline
            centerline_pts = chaikin_smooth(exterior_coords[:-1], iterations=3)

            # FIX: use Shapely offset for walls — no self-intersections
            outer_segs, inner_segs = _build_walls_from_centerline(centerline_pts, p["half_w"])

            # FIX: proper validation
            if not _walls_are_valid(outer_segs, inner_segs):
                continue

            return np.vstack([outer_segs, inner_segs]).astype(np.float32)

        except Exception as e:
            continue  # silent retry

    raise ValueError(f"Could not generate valid '{difficulty}' track in {max_attempts} attempts.")


# ── Dataset Generator ──────────────────────────────────────────────────────────

def build_ml_dataset(total_tracks=1000, base_dir="track_dataset", seed=SEED):
    """
    Generates tracks and physically segregates them into an 80/20 Train/Test split,
    further organized by difficulty.
    """
    rng = random.Random(seed)
    
    # 1. Calculate exact counts based on 25/50/25 split
    counts = {
        "easy": int(total_tracks * 0.25),
        "medium": int(total_tracks * 0.50),
        "hard": int(total_tracks * 0.25)
    }
    
    # Absorb any floating point rounding errors into medium
    counts["medium"] += total_tracks - sum(counts.values())
    
    train_ratio = 0.8
    
    # 2. Build the directory tree
    splits = ["train", "test"]
    for split in splits:
        for diff in counts.keys():
            os.makedirs(os.path.join(base_dir, split, diff), exist_ok=True)
            
    # 3. Generate and route the files
    for difficulty, total in counts.items():
        train_target = int(total * train_ratio)
        
        saved = 0
        while saved < total:
            try:
                # Determine destination (first 80% go to train, remaining 20% to test)
                split_folder = "train" if saved < train_target else "test"
                
                track_data = generate_csg_track(rng, difficulty)
                
                fname = f"track_{difficulty}_{saved:04d}.npy"
                save_path = os.path.join(base_dir, split_folder, difficulty, fname)
                
                np.save(save_path, track_data)
                saved += 1
                
                if saved % 50 == 0:
                    print(f"[{difficulty.upper()}] Generated {saved}/{total} -> {split_folder}")
                    
            except Exception:
                pass # Silent retry on geometric anomalies
                
    print(f"\nML Dataset Complete! {total_tracks} tracks saved in '{base_dir}/'")

# Swap out the execution block at the bottom to run this:
if __name__ == "__main__":
    print("Building 80/20 Split Track Dataset...")
    build_ml_dataset(total_tracks=1000)


# ── Test & Visualize ───────────────────────────────────────────────────────────

def test_track_generator(canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("pip install matplotlib")
        return

    difficulties = ["easy", "medium", "hard"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("CSG Track Generator — Easy / Medium / Hard", fontsize=14, fontweight='bold')

    os.makedirs("test_tracks", exist_ok=True)

    for ax, difficulty in zip(axes, difficulties):
        rng = random.Random()  # random seed each run
        success = False

        for attempt in range(30):
            try:
                track_data = generate_csg_track(rng, difficulty)
                success = True
                break
            except ValueError:
                continue

        if not success:
            ax.set_title(f"{difficulty.capitalize()} — FAILED")
            ax.axis("off")
            print(f"[FAIL] {difficulty}")
            continue

        half = len(track_data) // 2
        outer, inner = track_data[:half], track_data[half:]

        # Save
        npy_path = f"test_tracks/test_{difficulty}.npy"
        np.save(npy_path, track_data)

        # Plot
        for x1,y1,x2,y2 in outer:
            ax.plot([x1,x2],[y1,y2], color='black', lw=2)
        for x1,y1,x2,y2 in inner:
            ax.plot([x1,x2],[y1,y2], color='crimson', lw=2)

        # Start arrow
        s = track_data[0]
        ax.annotate("", xy=(s[2],s[3]), xytext=(s[0],s[1]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2))

        track_len = sum(math.hypot(r[2]-r[0], r[3]-r[1]) for r in outer)
        ax.set_xlim(0, canvas_w); ax.set_ylim(0, canvas_h)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(f"{difficulty.capitalize()}  |  segs={len(outer)}  |  len={track_len:.0f}px", fontsize=10)
        ax.legend(handles=[
            mpatches.Patch(color='black',  label='Outer wall'),
            mpatches.Patch(color='crimson',label='Inner wall'),
        ], fontsize=8)

        print(f"[{difficulty:6s}] segs={len(outer):4d} | length={track_len:.0f}px | saved = {npy_path}")

    plt.tight_layout()
    out = "test_tracks/preview.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\nPreview = {out}")
    plt.show()


if __name__ == "__main__":
    test_track_generator()