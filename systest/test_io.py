import cholla_utils
import numpy as np

from utils import SimTester


def test_restart(simtester: SimTester):
    param_file = "input/simple_sound_wave_restart.txt"
    simtester.configure_sim(name="initial-run", nproc=1, param_file=param_file)
    simtester.configure_sim(
        name="restarted-run",
        nproc=1,
        param_file=param_file,
        override_params={"init": "Read_Grid", "nfile": 1},
        set_indir_from_name_output="initial-run",
    )

    # run the sim(s)
    rslt = simtester.exec()

    # now lets do comparisons
    # -> we expect bitwise identical results
    orig_path = rslt["initial-run"].path / "2" / "2.h5.0"
    restart_path = rslt["restarted-run"].path / "2" / "2.h5.0"

    orig_field_list = cholla_utils.get_native_fields(orig_path)
    restart_field_list = cholla_utils.get_native_fields(restart_path)
    assert orig_field_list == restart_field_list

    orig_data = cholla_utils.load_field(orig_path, orig_field_list)
    restart_data = cholla_utils.load_field(restart_path, orig_field_list)

    for field_name in orig_field_list:
        np.testing.assert_equal(
            orig_data[field_name],
            restart_data[field_name],
            err_msg=(
                f"the {field_name!r} field should be bitwise identical (the presence "
                "of a restart shouldn't affect the result)"
            ),
        )
