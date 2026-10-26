"""Matched upper-limb joint frames for display; never used by the load solver."""
import numpy as np
import opensim as osim

class UpperBodyDisplay:
    def __init__(self, data, target_scale):
        self.model = osim.Model(str(data/'Models/Rajagopal_OpenSense/Rajagopal2015_opensense.osim'))
        self.state = self.model.initSystem()
        self.ratio = np.asarray(target_scale) / np.array([1., 1.05115, 1.26205])
        for side in ['r', 'l']:
            for name in ['arm_flex_', 'arm_add_', 'arm_rot_', 'elbow_flex_', 'pro_sup_', 'wrist_flex_', 'wrist_dev_']:
                self.model.getCoordinateSet().get(name+side).setLocked(self.state, False)

    def set_pose(self, hip_right, hip_left):
        for side, hip in [('r', hip_right), ('l', hip_left)]:
            values = {'arm_flex_': np.clip(-.55*hip, -.30, .30),
                      'arm_add_': np.deg2rad(-3), 'arm_rot_': 0.,
                      'elbow_flex_': np.deg2rad(18), 'pro_sup_': np.deg2rad(70),
                      'wrist_flex_': 0., 'wrist_dev_': 0.}
            for prefix, value in values.items():
                self.model.getCoordinateSet().get(prefix+side).setValue(self.state, float(value), False)
        self.model.realizePosition(self.state)

    def mesh_components(self):
        for component in self.model.getComponentsList():
            if component.getConcreteClassName() != 'Mesh': continue
            mesh = osim.Mesh.safeDownCast(component)
            body = mesh.getFrame().findBaseFrame().getName()
            if body.startswith(('humerus_', 'ulna_', 'radius_', 'hand_')) or mesh.get_mesh_file() == 'hat_ribs_scap.vtp':
                yield mesh

    def world_vertices(self, frame, vertices, target_rotation, target_position):
        from visualize import transform
        rotation, position = transform(frame, self.state)
        torso_rotation, torso_position = transform(self.model.getBodySet().get('torso'), self.state)
        source_world = vertices @ rotation.T + position
        local = (source_world - torso_position) @ torso_rotation
        return (local * self.ratio) @ target_rotation.T + target_position
