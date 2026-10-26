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
