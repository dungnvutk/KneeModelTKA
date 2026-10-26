import unittest
import numpy as np
from visualize import patella_vertices, shade

class DisplayTests(unittest.TestCase):
    def test_patella_rigid_transform_and_mirroring(self):
        vertices = np.array([[0., 0., 0.], [1., 2., 3.], [2., 1., 4.]]) / 100
        original = vertices.copy()
        for side in ['r', 'l']:
            for angle in [0., -.5, -1.2]:
                result = patella_vertices(vertices, side, np.array([1., 2., 3.]), np.eye(3), angle)
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_allclose(np.linalg.norm(result[1:] - result[0], axis=1),
                                           np.linalg.norm(vertices[1:] - vertices[0], axis=1))
        np.testing.assert_array_equal(vertices, original)

    def test_shading_degenerate_face_is_finite(self):
        colors = shade(np.zeros((3, 3)), np.array([[0, 1, 2]]), '#ffffff')
        self.assertTrue(np.isfinite(colors).all())
        self.assertTrue(((colors >= 0) & (colors <= 1)).all())


class ArmDisplayTests(unittest.TestCase):
    def test_joint_centers_stay_connected_through_swing(self):
        from arm_visual import UpperBodyDisplay
        from simulate import DATA
        from visualize import xyz
        upper = UpperBodyDisplay(DATA, [1., 1., 1.])
        for angle in [-.6, 0., .6]:
            upper.set_pose(angle, -angle)
            for side in ['r', 'l']:
                for prefix in ['acromial_', 'elbow_', 'radioulnar_', 'radius_hand_']:
                    joint = upper.model.getJointSet().get(prefix+side)
                    parent = xyz(joint.getParentFrame().getPositionInGround(upper.state))
                    child = xyz(joint.getChildFrame().getPositionInGround(upper.state))
                    np.testing.assert_allclose(parent, child, atol=1e-10)

    def test_torso_mapping_preserves_attachment(self):
        from arm_visual import UpperBodyDisplay
        from simulate import DATA
        upper = UpperBodyDisplay(DATA, [1., 1., 1.])
        upper.set_pose(0., 0.)
        torso = upper.model.getBodySet().get('torso')
        position = np.array([1., 2., 3.])
        actual = upper.world_vertices(torso, np.zeros((1,3)), np.eye(3), position)
        np.testing.assert_allclose(actual[0], position, atol=1e-10)
