"""Dark-theme anatomical replay. Patella meshes are illustrative, not dynamic bodies."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import opensim as osim

def xyz(v):
    return np.array([v.get(i) for i in range(3)])

def transform(frame, state):
    tr = frame.getTransformInGround(state)
    return np.array([[tr.R().get(i, j) for j in range(3)] for i in range(3)]), xyz(tr.p())

def shade(vertices, faces, color):
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
    light = np.array([.4, -.5, .8]); light /= np.linalg.norm(light)
    intensity = .48 + .52 * np.abs(normal @ light)
    return np.clip(intensity[:, None] * np.array(to_rgb(color)), 0, 1)

def patella_vertices(vertices, side, knee_center, femur_rotation, knee_angle):
    """Illustrative placement: half-flexion rotation, 40 mm anterior offset.
    
    This heuristic is for display only, not subject-specific patellar tracking.
    Mesh is centered; left side mirrors local z. No mass, constraints or forces.
    """
    local = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    local = local.copy()
    if side == 'l':
        local[:, 2] *= -1
    a = knee_angle / 2
    c, s = np.cos(a), np.sin(a)
    rotation = femur_rotation @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    center = knee_center + rotation @ np.array([.04, .015, 0])
    return local @ rotation.T + center

def render(model, export, out, fps, preview_only=False):
    from simulate import read_storage, read_vtp, DATA, PIPE
    motion = read_storage(PIPE / 'subject01_walk1_ik.mot')
    activation = read_storage(out / 'walking_StaticOptimization_activation.sto')
    state = model.initSystem()
    times = np.linspace(export.time_s.iloc[0], export.time_s.iloc[-1],
                        max(2, round((export.time_s.iloc[-1]-export.time_s.iloc[0])*fps)+1))
    bg, panel, text = '#0b1220', '#111e31', '#e6edf7'
    fig = plt.figure(figsize=(13, 7), facecolor=bg)
    ax = fig.add_axes([.01, .10, .53, .78], projection='3d', facecolor=bg)
    curve = fig.add_axes([.60, .24, .36, .54], facecolor=panel)
    ax.set_axis_off(); ax.view_init(elev=9, azim=-62)
    ax.set_box_aspect((1.15, .65, 1.85))
    meshes = []
    for component in model.getComponentsList():
        if component.getConcreteClassName() != 'Mesh':
            continue
        mesh = osim.Mesh.safeDownCast(component)
        if mesh.get_mesh_file() in ['treadmill.vtp', 'hat_ribs.vtp']:
            continue
        vertices, faces = read_vtp(DATA/'Geometry'/mesh.get_mesh_file())
        vertices *= xyz(mesh.get_scale_factors())
        poly = Poly3DCollection([], edgecolor='none')
        ax.add_collection3d(poly)
        meshes.append((mesh.getFrame(), vertices, faces, poly))
    from arm_visual import UpperBodyDisplay
    torso_scale = None
    for component in model.getComponentsList():
        if component.getConcreteClassName() == 'Mesh':
            mesh = osim.Mesh.safeDownCast(component)
            if mesh.get_mesh_file() == 'hat_spine.vtp':
                torso_scale = xyz(mesh.get_scale_factors())
    if torso_scale is None:
        raise ValueError('Torso geometry scale is missing')
    upper_body = UpperBodyDisplay(DATA, torso_scale)
    arms = []
    for mesh in upper_body.mesh_components():
        vertices, faces = read_vtp(DATA/'Geometry'/mesh.get_mesh_file())
        vertices *= xyz(mesh.get_scale_factors())
        poly = Poly3DCollection([], edgecolor='none')
        ax.add_collection3d(poly)
        arms.append((mesh.getFrame(), vertices, faces, poly))
    pv, pf = read_vtp(DATA/'Geometry/pat.vtp')
    patellas = {}
    for side in ['r', 'l']:
        poly = Poly3DCollection([], edgecolor='none')
        ax.add_collection3d(poly); patellas[side] = poly
    muscles = list(model.getMuscles())
    for muscle in muscles:
        if muscle.getName() not in activation:
            raise ValueError('Missing activation: '+muscle.getName())
    paths = Line3DCollection([], linewidths=2, alpha=.9)
    ax.add_collection3d(paths, autolim=False)
    floor = Poly3DCollection([], facecolor='#17273b', alpha=.65, edgecolor='#30415a')
    ax.add_collection3d(floor)
    for side, name, color in [('r', 'Right knee', '#60b9ff'), ('l', 'Left knee', '#ffb36b')]:
        curve.plot(export.time_s, export[f'{side}_resultant_BW'], label=name, color=color, lw=2.5)
    cursor = curve.axvline(times[0], color=text, lw=1, alpha=.7)
    curve.set_xlabel('Trial time (s)', color=text, labelpad=12)
    curve.set_ylabel('Resultant force / body weight', color=text, labelpad=10)
    curve.set_title('KNEE LOADS', color=text, loc='left', pad=18, fontsize=12, weight='bold')
    curve.tick_params(colors='#a7b8ce')
    for spine in curve.spines.values(): spine.set_color('#30415a')
    curve.grid(color='#344359', alpha=.5)
    curve.legend(facecolor=panel, edgecolor='none', labelcolor=text, loc='upper right')
    fig.text(.05, .94, 'WALKING  /  MUSCULOSKELETAL MODEL', color=text, fontsize=19, weight='bold')
    stamp = fig.text(.60, .84, '', color='#a7b8ce', fontsize=12)
    fig.text(.05, .06, 'Ivory: bones    Gold: visual patellae    Red → orange: increasing muscle activation',
             color=text, fontsize=10)
    fig.text(.05, .025, 'Recorded gait • muscle paths, not muscle volumes • arms and patellae illustrative • unvalidated load estimates',
             color='#91a3ba', fontsize=9)
    def update(t):
        state.setTime(float(t))
        for coord in model.getCoordinateSet():
            value = float(np.interp(t, motion.time, motion[coord.getName()]))
            if coord.getMotionType() == osim.Coordinate.Rotational: value = np.deg2rad(value)
            coord.setValue(state, value, False)
        model.realizePosition(state)
        for frame, vertices, faces, poly in meshes:
            rot, pos = transform(frame, state)
            world = (vertices @ rot.T + pos)[:, [0, 2, 1]]
            poly.set_verts(world[faces]); poly.set_facecolor(shade(world, faces, '#e3e8eb'))
        torso_rot, torso_pos = transform(model.getBodySet().get('torso'), state)
        upper_body.set_pose(*(model.getCoordinateSet().get('hip_flexion_'+side).getValue(state)
                              for side in ['r', 'l']))
        for frame, vertices, faces, poly in arms:
            world = upper_body.world_vertices(frame, vertices, torso_rot, torso_pos)[:, [0, 2, 1]]
            poly.set_verts(world[faces]); poly.set_facecolor(shade(world, faces, '#e3e8eb'))
        for side, poly in patellas.items():
            rot, _ = transform(model.getBodySet().get('femur_'+side), state)
            _, knee = transform(model.getBodySet().get('tibia_'+side), state)
            angle = model.getCoordinateSet().get('knee_angle_'+side).getValue(state)
            world = patella_vertices(pv, side, knee, rot, angle)[:, [0, 2, 1]]
            poly.set_verts(world[pf]); poly.set_facecolor(shade(world, pf, '#ffd272'))
        segments, colors, widths = [], [], []
        for muscle in muscles:
            path = muscle.getGeometryPath().getCurrentPath(state)
            points = np.array([xyz(path.get(i).getLocationInGround(state)) for i in range(path.getSize())])
            if len(points) < 2: continue
            segments.append(points[:, [0, 2, 1]])
            a = np.clip(np.interp(t, activation.time, activation[muscle.getName()]), 0, 1)
            colors.append((.65 + .35*a, .12 + .45*a, .20 - .08*a))
            widths.append(1.1 + 1.8*np.sqrt(muscle.getMaxIsometricForce()/5000))
        paths.set_segments(segments); paths.set_color(colors); paths.set_linewidth(widths)
        center = float(np.interp(t, motion.time, motion.pelvis_tx))
        ax.set_xlim(center-.58, center+.58); ax.set_ylim(-.34, .34); ax.set_zlim(-.02, 1.88)
        floor.set_verts([[(center-.65,-.32,0), (center+.65,-.32,0),
                          (center+.65,.32,0),(center-.65,.32,0)]])
        cursor.set_xdata([t, t]); stamp.set_text(f'{t:.2f} s    |    92 muscle paths')
        return []
    update(times[len(times)//2])
    fig.savefig(out/'walking_preview.png', dpi=150, facecolor=bg)
    if not preview_only:
        animation = FuncAnimation(fig, update, frames=times, interval=1000/fps)
        animation.save(out/'walking.gif', writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)

if __name__ == '__main__':
    import argparse
    import pandas as pd
    from simulate import ROOT, DATA, fetch_assets
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'results')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--preview-only', action='store_true')
    args = p.parse_args()
    if not 1 <= args.fps <= 60: p.error('fps must be between 1 and 60')
    fetch_assets()
    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(DATA/'Geometry'))
    render(osim.Model(str(args.output/'model.osim')),
           pd.read_csv(args.output/'knee_forces.csv'), args.output, args.fps, args.preview_only)
