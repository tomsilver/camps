"""Settings used throughout the directory.
"""

import time  # pylint:disable=unused-import


class EnvConfig:
    """Environment-specific constants.
    """
    family_to_run = "tampnamo"
    # family_to_run = "tampbins"
    # family_to_run = "enemiesbig"
    # family_to_run = "enemiessmall"
    train_tests = {  # dict from family to 1) list of train envs, 2) test env
        "tampnamo": {"train": [],
                     "test": []},
        "tampbins": {"train": [],
                     "test": []},
        "enemiesbig": {"train": [],
                       "test": []},
        "enemiessmall": {"train": [],
                         "test": []},
    }
    for diff in ["Easy", "Medium", "Hard"]:
        for ind in range(50):
            train_tests["enemiesbig"]["train"].append(
                "EnemiesEnvFamilyBig{}{}".format(diff, ind))
            train_tests["enemiessmall"]["train"].append(
                "EnemiesEnvFamilySmall{}{}".format(diff, ind))
        for ind in range(60, 70):
            train_tests["enemiesbig"]["test"].append(
                "EnemiesEnvFamilyBig{}{}".format(diff, ind))
            train_tests["enemiessmall"]["test"].append(
                "EnemiesEnvFamilySmall{}{}".format(diff, ind))
    for ind in range(50):
        train_tests["tampnamo"]["train"].append(
            "TAMPNAMOEnvFamily{}".format(ind))
    for ind in range(60, 70):
        train_tests["tampnamo"]["test"].append(
            "TAMPNAMOEnvFamily{}".format(ind))
    for ind in range(50):
        train_tests["tampbins"]["train"].append(
            "TAMPBinsEnvFamily{}".format(ind))
    for ind in range(60, 70):
        train_tests["tampbins"]["test"].append(
            "TAMPBinsEnvFamily{}".format(ind))
    if family_to_run == "tampnamo":
        max_episode_length = 20
        gamma = 0.99
        consider_conj_constraints = False
        consider_disj_constraints = True
        max_constraint_length = 8
        max_constraint_domain_size = 20
        indep_check_num_states = 25
        indep_check_num_changes_action = 25
        indep_check_num_changes_notaction = 25
        net_arch = "CNN"
        train_solver_timeout = 10
        test_solver_timeout = 60
        loss_thresh = 0.001
    elif family_to_run == "tampbins":
        max_episode_length = 20
        gamma = 0.99
        consider_conj_constraints = True
        consider_disj_constraints = True
        max_constraint_length = 2
        max_constraint_domain_size = 20
        indep_check_num_states = 25
        indep_check_num_changes_action = 25
        indep_check_num_changes_notaction = 25
        net_arch = "FCN"
        train_solver_timeout = 20
        test_solver_timeout = 60
        loss_thresh = 0.001
    elif family_to_run == "enemiesbig":
        max_episode_length = 25
        gamma = 0.99
        consider_conj_constraints = False
        consider_disj_constraints = True
        max_constraint_length = 4
        max_constraint_domain_size = 4
        indep_check_num_states = 100
        indep_check_num_changes_action = 50
        indep_check_num_changes_notaction = 50
        net_arch = "CNN"
        train_solver_timeout = 5
        test_solver_timeout = 10
        loss_thresh = 0.5
    elif family_to_run == "enemiessmall":
        max_episode_length = 25
        gamma = 0.99
        consider_conj_constraints = False
        consider_disj_constraints = True
        max_constraint_length = 3
        max_constraint_domain_size = 3
        indep_check_num_states = 100
        indep_check_num_changes_action = 50
        indep_check_num_changes_notaction = 50
        net_arch = "CNN"
        train_solver_timeout = 5
        test_solver_timeout = 10
        loss_thresh = 0.5


class SolverConfig:
    """Solver-specific constants.
    """
    # solver_name = "ValueIteration"
    # solver_name = "MCTS"
    solver_name = "TAMPSolver"
    # solver_name = "TAMPSolverStilman"
    # solver_name = "BFSReplan"
    if solver_name == "ValueIteration":
        vi_epsilon = 0.01
        vi_maxiters = 1000
    elif solver_name == "AsyncValueIteration":
        avi_queue_size = 1000
        avi_queue_epsilon = 0.01
        avi_maxiters = 10000
    elif solver_name == "MCTS":
        mcts_c = 500
        mcts_timelimit = 0.25
    # RRT stuff, not in an elif b/c needed by TAMP regardless of solver.
    extend_granularity = 0.05
    birrt_num_attempts = 10
    birrt_num_iters = 50
    birrt_smooth_amt = 50


class ApproachConfig:
    """Approach-specific constants.
    """
    # approach_names is a list of classes from approaches/
    approach_names = [
        "ModelBased",
        # "ModelBasedNoDropping",
        # "FullTestPlanner",
        # "PlanTransfer",
        # "PolicyTransfer",
        # "TaskConditionedPolicyTransfer",
    ]
    model_path_prefix = "trained_models/{}_{}".format(
        EnvConfig.family_to_run, SolverConfig.solver_name)
    limbo_reward = -100
    lam_dict = {  # reward vs. time tradeoff coefficient
        "ValueIteration": 1,
        "MCTS": 0,
        "TAMPSolver": 10,
        "BFSReplan": 100,
    }
    try:
        lam = lam_dict[SolverConfig.solver_name]
    except KeyError:
        lam = lam_dict[SolverConfig.solver_name+EnvConfig.family_to_run]
    retro_num_trajs = 1 # Number of trajectories to collect for each policy


class GeneralConfig:
    """General configuration constants.
    """
    mode = "both"  # "train" or "test" or "both"
    total_num_runs = 10
    if mode == "train":
        total_num_runs = 1
    if EnvConfig.family_to_run == "tampbins":
        num_eval_episodes_train = 3
        num_eval_episodes_test = 5
    elif EnvConfig.family_to_run.startswith("tamp") or \
         EnvConfig.family_to_run.startswith("enemies"):
        num_eval_episodes_train = 5
        num_eval_episodes_test = 10
    max_satisfy_tries = 1000000
    start_seed = int(time.time())
    seed = None
    rand_state = None
    verbosity = 1
    do_render = False
    do_training_render = False
    use_debug_info_visualizer = False


def print_settings():
    """Print these settings.
    """
    print("Seed: {}".format(GeneralConfig.seed))
    print("Task family: {}".format(EnvConfig.family_to_run))
    print("Solver name: {}".format(SolverConfig.solver_name))
    print("Approaches: {}".format(ApproachConfig.approach_names))
    print("Train/test split: {}".format(
        EnvConfig.train_tests[EnvConfig.family_to_run]))
    print()
