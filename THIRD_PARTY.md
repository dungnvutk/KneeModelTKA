# OpenSim asset attribution

Assets are downloaded from opensim-org/opensim-models at the commit pinned in assets.json. They remain the work of their original authors; this project does not relicense them.

The subject01 model's embedded credits identify Delp S.L., Loan J.P., Hoy M.G., Zajac F.E., Topp E.L., Rosen J.M., Thelen D.G., Anderson F.C., and Seth A. The 3D 23-DOF gait model was created by D.G. Thelen (University of Wisconsin–Madison), Ajay Seth, Frank C. Anderson and Scott L. Delp (Stanford University).

Lower-extremity joints follow Delp et al. (1990); low-back joint/anthropometry follow Anderson and Pandy (1999, 2001); the planar knee follows Yamaguchi and Zajac (1989). The model credits specify **Creative Commons Attribution 3.0**, https://creativecommons.org/licenses/by/3.0/ . The model includes moving quadriceps path points and removes the explicit patella.

Walking motion, force-plate data and skeletal meshes are distributed with the official OpenSim examples. Preserve their upstream notices and consult upstream for asset-specific terms. Generated example renderings are derived from these assets and carry the same attribution.

Changes made here: addition of coordinate reserve actuators, generated analysis settings, static optimization/joint reaction computation, and Python plots/animation. The original downloaded files are unmodified and hash-checked.

The anatomy display additionally downloads OpenSim's `Geometry/pat.vtp` from the same pinned source. It is centered, mirrored for the left side, and positioned using an illustrative display-only rule. The original downloaded mesh remains unmodified. Muscle paths are evaluated from the existing Gait2392 model.
