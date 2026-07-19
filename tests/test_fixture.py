"""The bundled synthetic fixture loads and exercises the checkpoints."""
import numpy as np
from pose_dynamics import load_pose_csv
from pose_dynamics.data import example_fixture, generate_fixture
from pose_dynamics.preprocessing import mask_low_confidence, interpolate_gaps, butterworth_filter


def test_fixture_exists_and_loads():
    seq = load_pose_csv(example_fixture(), frame_rate=60.0)
    assert seq.dims == 2 and seq.has_confidence
    assert seq.n_keypoints == 6 and seq.n_frames == 5400


def test_fixture_has_maskable_gaps_and_a_long_one():
    seq = load_pose_csv(example_fixture(), frame_rate=60.0)
    masked = mask_low_confidence(seq, 0.30)
    assert masked.missing_fraction() > 0        # confidence dropouts exist
    interp = interpolate_gaps(masked, 60)
    # a long gap (>60 frames) must survive interpolation
    assert interp.missing_fraction() > 0
    # filtering preserves the surviving gaps
    filt = butterworth_filter(interp, 10.0, 4)
    assert np.array_equal(~np.isfinite(interp.coords), ~np.isfinite(filt.coords))


def test_fixture_is_deterministic():
    a = generate_fixture(seed=0)
    b = generate_fixture(seed=0)
    assert a.equals(b)


def test_plot_keypoints_2d_and_3d():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from pose_dynamics.data import PoseSequence
    s2 = PoseSequence(coords=np.random.default_rng(0).normal(size=(20, 5, 2)),
                      keypoint_names=[f"k{i}" for i in range(5)], frame_rate=30.0)
    ax = s2.plot_keypoints()
    assert ax.has_data()
    s3 = PoseSequence(coords=np.random.default_rng(1).normal(size=(20, 5, 3)),
                      keypoint_names=[f"k{i}" for i in range(5)], frame_rate=30.0)
    ax3 = s3.plot_keypoints()
    assert len(ax3.figure.axes) == 2   # front + top projections


def test_cli_inspect_saves_plot(tmp_path):
    from pose_dynamics.__main__ import main
    from pose_dynamics.data import example_fixture
    out = tmp_path / "kp.png"
    assert main(["inspect", str(example_fixture()), "-o", str(out)]) == 0
    assert out.exists()
