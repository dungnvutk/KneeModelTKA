"""Reproducible OpenSim walking replay, static optimization and knee reactions."""
from __future__ import annotations
import os
# Avoid excessive BLAS threading in the frame-by-frame optimizer.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import opensim as osim

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
PIPE = DATA / 'Pipelines/Gait2392_Simbody'


def blob_hash(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def fetch_assets():
    manifest = json.loads((ROOT / 'assets.json').read_text())
    def fetch(item):
        dest = DATA / item['path']
        if dest.exists() and blob_hash(dest.read_bytes()) == item['git_blob_sha1']:
            return
        url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['commit']}/{item['path']}"
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        if blob_hash(data) != item['git_blob_sha1']:
            raise ValueError(f'Checksum mismatch: {item["path"]}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(dest.suffix + '.tmp')
        temp.write_bytes(data)
        temp.replace(dest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, manifest['files']))
    return manifest


def read_storage(path):
    lines = Path(path).read_text().splitlines()
    end = next(i for i, line in enumerate(lines) if line.strip().lower() == 'endheader')
    df = pd.read_csv(path, sep=r'\s+', skiprows=end + 1)
    if not np.isfinite(df.to_numpy()).all() or not np.all(np.diff(df.time) > 0):
        raise ValueError(f'Nonfinite data or nonincreasing times: {path}')
    return df


def write_storage(df, path):
    with open(path, 'w') as f:
        f.write(f'Actuator forces\nversion=1\nnRows={len(df)}\nnColumns={len(df.columns)}\ninDegrees=no\nendheader\n')
        df.to_csv(f, sep='\t', index=False, float_format='%.12g')


def strings(values):
    a = osim.ArrayStr()
    for value in values:
        a.append(value)
    return a


def prepare_model(out):
    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(DATA / 'Geometry'))
    model = osim.Model(str(PIPE / 'OutputReference/subject01_simbody.osim'))
    # Explicit reserves keep inverse dynamics solvable; their magnitudes are
    # exported, never silently interpreted as biological muscle forces.
    for coordinate in model.getCoordinateSet():
        if not coordinate.get_locked():
            a = osim.CoordinateActuator(coordinate.getName())
            a.setName('reserve_' + coordinate.getName())
            a.setOptimalForce(100 if coordinate.getName().startswith('pelvis_') else 10)
            a.setMinControl(-float('inf'))
            a.setMaxControl(float('inf'))
            model.addForce(a)
    model.printToXML(str(out / 'model.osim'))
    loads = osim.ExternalLoads(str(PIPE / 'subject01_walk1_grf.xml'), True)
    loads.setDataFileName(str(PIPE / 'subject01_walk1_grf.mot'))
    loads.printToXML(str(out / 'loads.xml'))
    return model


def analyze(model, out, start, end):
    tool = osim.AnalyzeTool()
    tool.setName('walking')
    tool.setModelFilename(str(out / 'model.osim'))
    tool.setCoordinatesFileName(str(PIPE / 'subject01_walk1_ik.mot'))
    tool.setExternalLoadsFileName(str(out / 'loads.xml'))
    tool.setInitialTime(start)
    tool.setFinalTime(end)
    tool.setLowpassCutoffFrequency(6)
    tool.setResultsDir(str(out))
    so = osim.StaticOptimization()
    so.setUseModelForceSet(True)
    so.setUseMusclePhysiology(True)
    so.setActivationExponent(2)
    so.setMaxIterations(200)
    so.setConvergenceCriterion(1e-4)
    tool.updAnalysisSet().cloneAndAppend(so)
    tool.printToXML(str(out / 'static_optimization.xml'))
    if not osim.AnalyzeTool(str(out / 'static_optimization.xml')).run():
        raise RuntimeError('Static optimization failed; inspect opensim.log')
    log = (out / 'opensim.log').read_text()
    if 'OPTIMIZATION FAILED' in log or 'could not find a solution' in log:
        raise RuntimeError('At least one frame did not converge; no force plot produced.')
    forces = read_storage(out / 'walking_StaticOptimization_force.sto')
    # SO also writes external loads. JointReaction needs actuator columns only;
    # ground forces are supplied separately by the identical ExternalLoads file.
    actuator_names = [a.getName() for a in model.getActuators()]
    write_storage(forces[['time'] + actuator_names], out / 'actuator_forces.sto')
    tool.updAnalysisSet().clearAndDestroy()
    for name, body, frame in [('KneeReaction', 'parent', 'parent'),
                              ('ParentGround', 'parent', 'ground'),
                              ('ChildGround', 'child', 'ground')]:
        reaction = osim.JointReaction()
        reaction.setName(name)
        reaction.setJointNames(strings(['knee_r', 'knee_l']))
        reaction.setOnBody(strings([body]))
        reaction.setInFrame(strings([frame]))
        reaction.setForcesFileName(str(out / 'actuator_forces.sto'))
        tool.updAnalysisSet().cloneAndAppend(reaction)
    for name, body, frame in [('HipReaction', 'child', 'child'),
                              ('HipParentGround', 'parent', 'ground'),
                              ('HipChildGround', 'child', 'ground')]:
        reaction = osim.JointReaction()
        reaction.setName(name)
        reaction.setJointNames(strings(['hip_r', 'hip_l']))
        reaction.setOnBody(strings([body]))
        reaction.setInFrame(strings([frame]))
        reaction.setForcesFileName(str(out / 'actuator_forces.sto'))
        tool.updAnalysisSet().cloneAndAppend(reaction)
    tool.printToXML(str(out / 'joint_reaction.xml'))
    if not osim.AnalyzeTool(str(out / 'joint_reaction.xml')).run():
        raise RuntimeError('Joint reaction analysis failed')
    return forces


def force_report(model, forces, out, manifest):
    df = read_storage(out / 'walking_KneeReaction_ReactionLoads.sto')
    parent = read_storage(out / 'walking_ParentGround_ReactionLoads.sto')
    child = read_storage(out / 'walking_ChildGround_ReactionLoads.sto')
    if len(df) != len(forces) or not np.allclose(df.time, forces.time):
        raise ValueError('Force and kinematic sample times do not match')
    state = model.initSystem()
    mass = model.getTotalMass(state)
    weight = mass * np.linalg.norm([model.getGravity().get(i) for i in range(3)])
    export = pd.DataFrame({'time_s': df.time})
    summary = {'model_mass_kg': mass, 'body_weight_N': weight,
               'opensim_version': osim.GetVersion(), 'source_commit': manifest['commit'],
               'analysis': 'Prescribed walking + static optimization + joint reaction',
               'frame': 'Force on femur, expressed in ipsilateral femur frame',
               'clinical_validation': False, 'peaks': {}}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
    for row, side in enumerate(['r', 'l']):
        prefix = f'knee_{side}_on_femur_{side}_in_femur_{side}'
        xyz = df[[prefix + '_f' + a for a in 'xyz']].to_numpy()
        pg = parent[[f'knee_{side}_on_femur_{side}_in_ground_f{a}' for a in 'xyz']].to_numpy()
        cg = child[[f'knee_{side}_on_tibia_{side}_in_ground_f{a}' for a in 'xyz']].to_numpy()
        if not np.allclose(pg, -cg, atol=1e-5):
            raise ValueError('Parent/child action-reaction check failed')
        resultant = np.linalg.norm(xyz, axis=1)
        if not np.allclose(resultant, np.linalg.norm(pg, axis=1), atol=1e-5):
            raise ValueError('Frame-rotation norm invariance failed')
        for k, a in enumerate('xyz'):
            export[f'{side}_F{a}_N'] = xyz[:, k]
            axes[row, 0].plot(df.time, xyz[:, k], label=f'F{a}')
        export[f'{side}_resultant_N'] = resultant
        export[f'{side}_resultant_BW'] = resultant / weight
        axes[row, 1].plot(df.time, resultant / weight, color='#174ea6', lw=2)
        side_name = 'Right' if side == 'r' else 'Left'
        axes[row, 0].set_title(f'{side_name} knee — force on femur')
        axes[row, 1].set_title(f'{side_name} knee — resultant')
        axes[row, 0].set_ylabel('Force (N)')
        axes[row, 1].set_ylabel('Body weights')
        axes[row, 0].legend(ncol=3)
        summary['peaks'][side_name] = {'resultant_N': float(resultant.max()),
                                     'resultant_BW': float(resultant.max() / weight)}
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel('Trial time (s)')
    fig.suptitle('Gait2392 walking | model estimates, not implant contact pressures', fontsize=14)
    fig.savefig(out / 'knee_forces.png', dpi=180)
    plt.close(fig)
    export.to_csv(out / 'knee_forces.csv', index=False)
    reserve = forces[['time'] + [c for c in forces if c.startswith('reserve_')]]
    reserve.to_csv(out / 'reserve_forces.csv', index=False)
    summary['peak_abs_reserves'] = reserve.drop(columns='time').abs().max().to_dict()
    # Conservative diagnostic flags, not a substitute for validation.
    pelvis = reserve[[f'reserve_pelvis_t{a}' for a in 'xyz']].to_numpy()
    summary['peak_pelvis_residual_force_BW'] = float(np.linalg.norm(pelvis, axis=1).max() / weight)
    summary['quality_flags'] = []
    if summary['peak_pelvis_residual_force_BW'] > .05:
        summary['quality_flags'].append('Pelvis residual force exceeds 5% BW; improve dynamic consistency with RRA before research interpretation.')
    for side in ['r', 'l']:
        if reserve[f'reserve_knee_angle_{side}'].abs().max() > 2:
            summary['quality_flags'].append(f'{side}: knee reserve exceeds 2 Nm; assess reserve sensitivity.')
    summary['verification'] = ['finite and monotonic outputs', 'matching SO/JR sample times',
                               'parent/child force balance', 'rotation-invariant force magnitude']
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return export, summary


def hip_report(out, summary):
    """Report hip reactions on the femur, in its local frame, in N and BW."""
    df = read_storage(out / 'walking_HipReaction_ReactionLoads.sto')
    parent = read_storage(out / 'walking_HipParentGround_ReactionLoads.sto')
    child = read_storage(out / 'walking_HipChildGround_ReactionLoads.sto')
    knee = read_storage(out / 'walking_KneeReaction_ReactionLoads.sto')
    for other in [parent, child, knee]:
        if len(df) != len(other) or not np.allclose(df.time, other.time):
            raise ValueError('Hip reaction sample times do not match')
    export = pd.DataFrame({'time_s': df.time})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
    summary['hip_peaks'] = {}
    for row, side in enumerate(['r', 'l']):
        xyz = df[[f'hip_{side}_on_femur_{side}_in_femur_{side}_f{a}' for a in 'xyz']].to_numpy()
        pg = parent[[f'hip_{side}_on_pelvis_in_ground_f{a}' for a in 'xyz']].to_numpy()
        cg = child[[f'hip_{side}_on_femur_{side}_in_ground_f{a}' for a in 'xyz']].to_numpy()
        resultant = np.linalg.norm(xyz, axis=1)
        if not np.allclose(pg, -cg, atol=1e-5):
            raise ValueError('Hip parent/child action-reaction check failed')
        if not np.allclose(resultant, np.linalg.norm(cg, axis=1), atol=1e-5):
            raise ValueError('Hip frame-rotation norm invariance failed')
        for k, a in enumerate('xyz'):
            export[f'{side}_F{a}_N'] = xyz[:, k]
            axes[row, 0].plot(df.time, xyz[:, k], label=f'F{a}')
        export[f'{side}_resultant_N'] = resultant
        export[f'{side}_resultant_BW'] = resultant / summary['body_weight_N']
        axes[row, 1].plot(df.time, export[f'{side}_resultant_BW'], color='#174ea6', lw=2)
        name = 'Right' if side == 'r' else 'Left'
        axes[row, 0].set_title(f'{name} hip — force on femur')
        axes[row, 1].set_title(f'{name} hip — resultant')
        axes[row, 0].set_ylabel('Force (N)')
        axes[row, 1].set_ylabel('Body weights')
        axes[row, 0].legend(ncol=3)
        summary['hip_peaks'][name] = {'resultant_N': float(resultant.max()),
                                    'resultant_BW': float(export[f'{side}_resultant_BW'].max())}
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.set_xlabel('Trial time (s)')
    fig.suptitle('Gait2392 walking | hip joint reactions — generic model estimates')
    fig.savefig(out / 'hip_forces.png', dpi=180); plt.close(fig)
    export.to_csv(out / 'hip_forces.csv', index=False)
    summary['hip_verification'] = ['matching hip/knee times', 'parent/child force balance',
                                   'rotation-invariant force magnitude']
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return export


def read_vtp(path):
    root = ET.parse(path).getroot()
    points = root.find('.//Points/DataArray')
    arrays = root.findall('.//Polys/DataArray')
    if points.get('format') != 'ascii' or any(a.get('format') != 'ascii' for a in arrays):
        raise ValueError('This reader supports ASCII VTP assets only')
    vertices = np.fromstring(points.text, sep=' ').reshape(-1, 3)
    conn = next(a for a in arrays if a.get('Name') == 'connectivity')
    offs = next(a for a in arrays if a.get('Name') == 'offsets')
    indices = np.fromstring(conn.text, sep=' ', dtype=int)
    offsets = np.fromstring(offs.text, sep=' ', dtype=int)
    triangles, start = [], 0
    for stop in offsets:
        face = indices[start:stop]
        for j in range(1, len(face) - 1):
            triangles.append([face[0], face[j], face[j + 1]])
        start = stop
    return vertices, np.asarray(triangles)


def animate(model, export, out, fps):
    from visualize import render
    render(model, export, out, fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--start', type=float, default=.6)
    parser.add_argument('--end', type=float, default=1.8)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--no-animation', action='store_true')
    args = parser.parse_args()
    if not (.5 < args.start < args.end < 4) or args.fps < 1 or args.fps > 60:
        parser.error('Use 0.5 < start < end < 4.0 seconds, and 1 <= fps <= 60')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    log = out / 'opensim.log'
    if log.exists(): log.unlink()
    osim.Logger.addFileSink(str(log))
    manifest = fetch_assets()
    model = prepare_model(out)
    forces = analyze(model, out, args.start, args.end)
    export, summary = force_report(model, forces, out, manifest)
    hip_report(out, summary)
    if not args.no_animation:
        animate(model, export, out, args.fps)
    print(json.dumps(summary, indent=2))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
