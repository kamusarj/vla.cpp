# ALOHA simulation dependencies

The simulator uses
[`google-deepmind/aloha_sim`](https://github.com/google-deepmind/aloha_sim) at
commit `d02904607cca1bf6dfb72f30b522506ac7ca0f91`, installed at setup time rather
than copied into this repository. It supplies the official bimanual ALOHA
robot, MuJoCo assets, cameras, contacts, and `dm_control.composer` dynamics.

`official_carrot.py` is adapted from the local Octo ALOHA simulation reference
under the MIT license reproduced in `OCTO_LICENSE`. `aloha_env.py` is the
vla.cpp adapter that preserves TurboVLA's two-image, left-arm 7-D contract.
