# Panda hand

`hand.stl` and `finger_0.obj` are copied unmodified from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), `franka_emika_panda/assets/`,
commit `370c43965323c0944b433b202aac5058b5f907a9` (Apache-2.0, see `LICENSE`).

Only the two meshes that Menagerie's `hand.xml` uses for collision are kept; the visual meshes are not
needed for the headless evaluation. `hand.xml` is our own model built around them (lift axis, force-limited fingers); it is loaded and
parameterised by `graspgen/sim/gripper.py`.
