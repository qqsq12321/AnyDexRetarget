"""Robot-specific link and URDF configurations for supported hands.

Each entry maps a robot type name to a dict containing:
- origin_link: Base/palm link name
- tip_links: Fingertip link names
- link1_names: Proximal phalanx link names
- link3_names: Middle/PIP link names
- link4_names: Distal/DIP link names
- urdf_subdir, urdf_file: URDF asset paths
- mjcf_subdir, mjcf_file: MuJoCo XML asset paths (optional)
- num_fingers: Number of fingers (4 or 5)
- neutral_qpos: Optional neutral joint position
- Offsets: link3_offsets, tip_offsets, link4_offsets (optional)
"""

ROBOT_CONFIGS = {
    # Shadow Hand (MuJoCo Menagerie style, rh_/lh_ prefix) - high quality meshes
    # Uses custom URDF that exactly matches MuJoCo Menagerie joint axes
    'shadow_hand': {
        'origin_link': 'rh_palm',  # Will be lh_palm for left hand
        'tip_links': ['rh_thtip', 'rh_fftip', 'rh_mftip', 'rh_rftip', 'rh_lftip'],
        'link1_names': ['rh_thproximal', 'rh_ffproximal', 'rh_mfproximal', 'rh_rfproximal', 'rh_lfproximal'],
        'link3_names': ['rh_thmiddle', 'rh_ffmiddle', 'rh_mfmiddle', 'rh_rfmiddle', 'rh_lfmiddle'],
        'link4_names': ['rh_thdistal', 'rh_ffdistal', 'rh_mfdistal', 'rh_rfdistal', 'rh_lfdistal'],
        'urdf_subdir': 'assets/shadow_hand',
        'urdf_file': {'right': 'right_hand_mj.urdf', 'left': 'left_hand_mj.urdf'},
        'mjcf_subdir': 'assets/shadow_hand',
        'mjcf_file': {'right': 'scene_right.xml', 'left': 'scene_left.xml'},
        'num_fingers': 5,
    },
    # Wuji Hand (5 fingers x 4 joints = 20 DOF)
    'wuji_hand': {
        'origin_link': 'right_palm_link',
        'tip_links': ['right_finger1_tip_link', 'right_finger2_tip_link', 'right_finger3_tip_link', 'right_finger4_tip_link', 'right_finger5_tip_link'],
        'link1_names': ['right_finger1_link1', 'right_finger2_link1', 'right_finger3_link1', 'right_finger4_link1', 'right_finger5_link1'],
        'link3_names': ['right_finger1_link3', 'right_finger2_link3', 'right_finger3_link3', 'right_finger4_link3', 'right_finger5_link3'],
        'link4_names': ['right_finger1_link4', 'right_finger2_link4', 'right_finger3_link4', 'right_finger4_link4', 'right_finger5_link4'],
        'num_fingers': 5,
    },
    # Gaia Hand20 (5 fingers x 4 joints = 20 revolute joints)
    'gaia_hand20': {
        'origin_link': 'right_base_link',
        'tip_links': ['right_thumb_tip', 'right_index_tip', 'right_middle_tip', 'right_ring_tip', 'right_little_tip'],
        'link1_names': ['right_thumb_link_1', 'right_index_link_1', 'right_middle_link_1', 'right_ring_link_1', 'right_little_link_1'],
        'link3_names': ['right_thumb_link_3', 'right_index_link_3', 'right_middle_link_3', 'right_ring_link_3', 'right_little_link_3'],
        'link4_names': ['right_thumb_link_4', 'right_index_link_4', 'right_middle_link_4', 'right_ring_link_4', 'right_little_link_4'],
        'urdf_subdir': 'assets/gaia_hand20',
        'urdf_file': {
            'right': 'gaiahand20_right.urdf',
            'left': 'gaiahand20_left.urdf',
        },
        'mjcf_subdir': 'assets/gaia_hand20',
        'mjcf_file': {
            'right': 'gaiahand20_right_mujoco.xml',
            'left': 'gaiahand20_left_mujoco.xml',
        },
        'num_fingers': 5,
        'neutral_qpos': [0.0] * 20,
    },
    # Allegro Hand (4 fingers: thumb, index, middle, ring - no pinky)
    # Finger order: thumb (link_12~15), index (link_0~3), middle (link_4~7), ring (link_8~11)
    'allegro_hand': {
        'origin_link': 'base_link',
        'tip_links': ['link_15.0_tip', 'link_3.0_tip', 'link_7.0_tip', 'link_11.0_tip'],
        'link1_names': ['link_12.0', 'link_0.0', 'link_4.0', 'link_8.0'],
        'link3_names': ['link_13.0', 'link_1.0', 'link_5.0', 'link_9.0'],
        'link4_names': ['link_14.0', 'link_2.0', 'link_6.0', 'link_10.0'],
        'num_fingers': 4,
    },
    # Inspire Hand (5 fingers, 2-DOF per non-thumb finger)
    # Non-thumb: proximal -> intermediate -> tip(fixed), so link3=proximal(PIP), link4=intermediate(DIP)
    'inspire_hand': {
        'origin_link': 'hand_base_link',
        'tip_links': ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'],
        'link1_names': ['thumb_proximal', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal'],
        'link3_names': ['thumb_proximal', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal'],
        'link4_names': ['thumb_intermediate', 'index_intermediate', 'middle_intermediate', 'ring_intermediate', 'pinky_intermediate'],
        'num_fingers': 5,
    },
    # Ability Hand (5 fingers, 2 links each)
    'ability_hand': {
        'origin_link': 'base',
        'tip_links': ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'],
        'link1_names': ['thumb_L1', 'index_L1', 'middle_L1', 'ring_L1', 'pinky_L1'],
        'link3_names': ['thumb_L1', 'index_L1', 'middle_L1', 'ring_L1', 'pinky_L1'],
        'link4_names': ['thumb_L2', 'index_L2', 'middle_L2', 'ring_L2', 'pinky_L2'],
        'num_fingers': 5,
    },
    # Leap Hand (4 fingers + thumb, no pinky)
    'leap_hand': {
        'origin_link': 'base',
        'tip_links': ['thumb_tip_head', 'index_tip_head', 'middle_tip_head', 'ring_tip_head'],
        'link1_names': ['thumb_pip', 'pip', 'pip_2', 'pip_3'],
        'link3_names': ['thumb_dip', 'dip', 'dip_2', 'dip_3'],
        'link4_names': ['thumb_fingertip', 'fingertip', 'fingertip_2', 'fingertip_3'],
        'num_fingers': 4,
    },
    # SVH Hand (5 fingers)
    'svh_hand': {
        'origin_link': 'right_hand_base_link',
        'tip_links': ['thtip', 'fftip', 'mftip', 'rftip', 'lftip'],
        'link1_names': ['right_hand_z', 'right_hand_l', 'right_hand_k', 'right_hand_j', 'right_hand_i'],
        'link3_names': ['right_hand_a', 'right_hand_p', 'right_hand_o', 'right_hand_n', 'right_hand_m'],
        'link4_names': ['right_hand_b', 'right_hand_t', 'right_hand_s', 'right_hand_r', 'right_hand_q'],
        'num_fingers': 5,
    },
    # LinkerHand L21
    # Virtual tip links added to URDF (fixed joints from last physical link)
    'linkerhand_l21': {
        'origin_link': 'hand_base_link',
        'tip_links': ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'],
        'link1_names': ['thumb_metacarpals', 'index_metacarpals', 'middle_metacarpals', 'ring_metacarpals', 'pinky_metacarpals'],
        'link3_names': ['thumb_metacarpals', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal'],
        'link4_names': ['thumb_distal', 'index_middle', 'middle_middle', 'ring_middle', 'pinky_middle'],
        'link3_offsets': [
            [0.018, 0.000, 0.000],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
        ],
        'tip_offsets': None,
        'link4_offsets': [
            [0.023, 0.000, 0.000],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
        ],
        'urdf_subdir': 'assets/linkerhand_l21',
        'urdf_file': {
            'right': 'right/linkerhand_l21_right_vis.urdf',
            'left': 'left/linkerhand_l21_left_vis.urdf',
        },
        'num_fingers': 5,
        'neutral_qpos': [0.0] * 17,
    },
    # Linker L6 (6 independent joints and 5 coupled PIP joints; simulation only)
    'linker_l6': {
        'origin_link': 'base_link',
        'tip_links': ['thumb_tip_link', 'index_tip_link', 'middle_tip_link', 'ring_tip_link', 'little_tip_link'],
        'link1_names': ['thumb_prox_base', 'index_prox', 'middle_prox', 'ring_prox', 'little_prox'],
        'link3_names': ['thumb_prox', 'index_prox', 'middle_prox', 'ring_prox', 'little_prox'],
        'link4_names': ['thumb_dist', 'index_dist', 'middle_dist', 'ring_dist', 'little_dist'],
        'urdf_subdir': 'assets/linker_l6',
        'urdf_file': {
            'right': 'right_linker/right_hand.urdf',
            'left': 'left_linker/left_hand.urdf',
        },
        'mjcf_subdir': 'assets/linker_l6',
        'mjcf_file': {
            'right': 'linker_l6_right_mujoco.xml',
            'left': 'linker_l6_left_mujoco.xml',
        },
        'num_fingers': 5,
        # Pinocchio order: index, little, middle, ring (MCP/PIP), then thumb.
        'neutral_qpos': {
            'right': [0.0] * 9 + [0.235968, 0.612611 + 0.611762 * 0.235968],
            'left': [0.0] * 9 + [0.235968, 0.611762 * 0.235968],
        },
    },
    # Linker L20 (GEORT L20 URDF, 21 revolute joints with 5 mimic joints)
    'linker_l20': {
        'origin_link': 'hand_base_link',
        'tip_links': [
            'thumb_distal',
            'index_distal',
            'middle_distal',
            'ring_distal',
            'pinky_distal',
        ],
        # The GEORT distal link frames are located at the DIP/IP joints.  These
        # local offsets place the targets at the visible fingertip surfaces.
        'tip_offsets': [
            [-0.0110, 0.0, 0.0277],
            [-0.0135, 0.0, 0.0188],
            [-0.0135, 0.0, 0.0188],
            [-0.0135, 0.0, 0.0188],
            [-0.0135, 0.0, 0.0188],
        ],
        'link1_names': [
            'thumb_metacarpals_base1',
            'index_metacarpals',
            'middle_metacarpals',
            'ring_metacarpals',
            'pinky_metacarpals',
        ],
        'link3_names': [
            'thumb_proximal',
            'index_middle',
            'middle_middle',
            'ring_middle',
            'pinky_middle',
        ],
        'link4_names': [
            'thumb_distal',
            'index_distal',
            'middle_distal',
            'ring_distal',
            'pinky_distal',
        ],
        'urdf_subdir': 'assets/linker_l20',
        'urdf_file': {
            'right': 'geort_right/linkerhand_l20_right.urdf',
            'left': 'geort_left/linkerhand_l20_left.urdf',
        },
        'num_fingers': 5,
        'neutral_qpos': [0.0] * 21,
    },
    # ROHand
    'rohand': {
        'origin_link': 'base_link',
        'tip_links': ['th_distal_link', 'if_distal_link', 'mf_distal_link', 'rf_distal_link', 'lf_distal_link'],
        'link1_names': ['th_root_link', 'if_slider_abpart_link', 'mf_slider_abpart_link', 'rf_slider_abpart_link', 'lf_slider_abpart_link'],
        'link3_names': ['th_root_link', 'if_slider_abpart_link', 'mf_slider_abpart_link', 'rf_slider_abpart_link', 'lf_slider_abpart_link'],
        'link4_names': ['th_proximal_link', 'if_proximal_link', 'mf_proximal_link', 'rf_proximal_link', 'lf_proximal_link'],
        'urdf_subdir': 'assets/rohand',
        'urdf_file': {
            'right': 'right/rohand_right_vis.urdf',
            'left': 'left/rohand_left_vis.urdf',
        },
        'num_fingers': 5,
    },
    # Unitree Dex5
    'unitree_dex5_hand': {
        'origin_link': 'base_link00',
        'tip_links': ['Link_14R', 'Link_24R', 'Link_34R', 'Link_44R', 'Link_54R'],
        'link1_names': ['Link_11R', 'Link_21R', 'Link_31R', 'Link_41R', 'Link_51R'],
        'link3_names': ['Link_12R', 'Link_22R', 'Link_32R', 'Link_42R', 'Link_52R'],
        'link4_names': ['Link_13R', 'Link_23R', 'Link_33R', 'Link_43R', 'Link_53R'],
        'urdf_subdir': 'assets/unitree_dex5_hand',
        'urdf_file': {
            'right': 'right/Dex5-URDF-R.urdf',
            'left': 'left/Dex5-URDF-L.urdf',
        },
        'num_fingers': 5,
        'neutral_qpos': [0.0] * 20,
    },
    # Sharpa Hand (5 fingers, 22 DOF: thumb 5, index/middle/ring 4 each, pinky 5)
    'sharpa_hand': {
        'origin_link': 'right_hand_C_MC',
        'tip_links': ['right_thumb_fingertip', 'right_index_fingertip', 'right_middle_fingertip', 'right_ring_fingertip', 'right_pinky_fingertip'],
        'link1_names': ['right_thumb_MC', 'right_index_PP', 'right_middle_PP', 'right_ring_PP', 'right_pinky_PP'],
        'link3_names': ['right_thumb_PP', 'right_index_MP', 'right_middle_MP', 'right_ring_MP', 'right_pinky_MP'],
        'link4_names': ['right_thumb_DP', 'right_index_DP', 'right_middle_DP', 'right_ring_DP', 'right_pinky_DP'],
        'urdf_subdir': 'assets/sharpa_hand',
        'urdf_file': {
            'right': 'right_sharpa_wave.urdf',
            'left': 'left_sharpa_wave.urdf',
        },
        'mjcf_subdir': 'assets/sharpa_hand',
        'mjcf_file': {
            'right': 'right_sharpa_wave.xml',
            'left': 'left_sharpa_wave.xml',
        },
        'num_fingers': 5,
        'neutral_qpos': [0.0] * 22,
    },
}
