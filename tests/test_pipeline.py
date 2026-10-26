import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from simulate import read_storage, write_storage, blob_hash, read_vtp

class PipelineTests(unittest.TestCase):
    def test_storage_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'forces.sto'
            expected = pd.DataFrame({'time': [0., .1, .2], 'force': [-2., 0., 3.]})
            write_storage(expected, path)
            np.testing.assert_allclose(read_storage(path), expected)

    def test_invalid_times_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.sto'
            write_storage(pd.DataFrame({'time': [1., 1.], 'force': [2., 3.]}), path)
            with self.assertRaises(ValueError): read_storage(path)

    def test_git_blob_checksum(self):
        self.assertEqual(blob_hash(b''), 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')

    def test_quad_mesh_triangulation(self):
        xml = '''<VTKFile><PolyData><Piece><Points><DataArray format="ascii">0 0 0 1 0 0 1 1 0 0 1 0</DataArray></Points><Polys><DataArray Name="connectivity" format="ascii">0 1 2 3</DataArray><DataArray Name="offsets" format="ascii">4</DataArray></Polys></Piece></PolyData></VTKFile>'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mesh.vtp';path.write_text(xml)
            vertices, faces = read_vtp(path)
            self.assertEqual(vertices.shape, (4, 3))
            np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])

if __name__ == '__main__': unittest.main()
