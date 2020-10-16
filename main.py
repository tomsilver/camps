"""Entry point.
"""

import argparse
import sys
import os
import time
import pickle
from collections import OrderedDict, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import approaches
import envs
import solvers
import utils
from settings import EnvConfig as ec
from settings import SolverConfig as sc
from settings import ApproachConfig as ac
from settings import GeneralConfig as gc
from settings import print_settings
np.set_printoptions(suppress=True, linewidth=np.inf)

APPROACH_TO_NAME = {
    "ModelBased": "CAMP (ours)",
    "ModelBasedNoDropping": "CAMP ablation",
    "FullTestPlanner": "Pure planning",
    "PlanTransfer": "Plan transfer",
    "PolicyTransfer": "Policy learning",
    "TaskConditionedPolicyTransfer": "Task-conditioned policy learning",
    "Stilman": "Stilman's (NAMO-specific)",
}

ENV_TO_NAME = {
    "enemiessmall": "Domain 1 (Gridworld), BFSReplan",
    "enemiesbig": "Domain 1 (Gridworld), MCTS",
    "tampnamo": "Domain 3 (Robotic NAMO)",
    "tampbins": "Domain 4 (Robotic Manipulation)",
    "dinner": "Domain 2 (Classical)",
}


class Runner:
    """Main runner class for multiple approaches and train/test splits.
    """
    def __init__(self):
        gc.rand_state = np.random.RandomState(seed=gc.start_seed)
        if not os.path.exists("plots/"):
            os.mkdir("plots/")
        if not os.path.exists("results/"):
            os.mkdir("results/")
        if not os.path.exists("trained_models/"):
            os.mkdir("trained_models/")
        self._single_runners = OrderedDict()
        train_env_names = ec.train_tests[ec.family_to_run]["train"]
        test_env_names = ec.train_tests[ec.family_to_run]["test"]
        solver = getattr(solvers, sc.solver_name)()
        for approach_name in ac.approach_names:
            runner_id = self._create_runner_id(approach_name)
            self._single_runners[runner_id] = SingleRunner(
                approach_name, solver, train_env_names, test_env_names)
        # runner_id = "tampnamo_TAMPSolverStilman_FullTestPlanner_10"
        # self._single_runners[runner_id] = SingleRunner(
        #     "FullTestPlanner", "TAMPSolver", train_env_names, test_env_names)

    @staticmethod
    def plot():
        """Only do plotting.
        """
        all_results = defaultdict(lambda: defaultdict(list))
        for i in range(gc.total_num_runs):
            for approach_name in ac.approach_names:
                runner_id = Runner._create_runner_id(approach_name)
                filename = "results/{}_{}.pkl".format(runner_id, i+1)
                with open(filename, "rb") as f:
                    result = pickle.load(f)
                for env_name, res in result.items():
                    all_results[runner_id][env_name].append(res)
            if ec.family_to_run == "tampnamo":
                runner_id = "tampnamo_TAMPSolverStilman_FullTestPlanner_10"
                filename = "results/{}_{}.pkl".format(runner_id, i+1)
                with open(filename, "rb") as f:
                    result = pickle.load(f)
                runner_id = "tampnamo_TAMPSolver_Stilman_10"
                for env_name, res in result.items():
                    all_results[runner_id][env_name].append(res)
            return_data = defaultdict(dict)
            plancost_data = defaultdict(dict)
            obj_value_data = defaultdict(dict)
            for runner_id, env_name_to_res in all_results.items():
                for env_name, res in env_name_to_res.items():
                    mean_res = np.mean(res, axis=0)
                    std_res = np.std(res, axis=0)
                    return_data[env_name][runner_id.split("_")[2]] = (
                        mean_res[0], std_res[0])
                    plancost_data[env_name][runner_id.split("_")[2]] = (
                        mean_res[1], std_res[1])
                    obj_value_data[env_name][runner_id.split("_")[2]] = (
                        mean_res[2], std_res[2])
        # Runner._plot(return_data, "Returns", "returns")
        # Runner._plot(plancost_data, "Solve Cost", "solvecosts")
        # Runner._plot(obj_value_data, "Objective Value", "objvalues")
        Runner._plot_2d(return_data, plancost_data)

    def run(self):
        """Main method; train and test each approach.
        """
        all_results = defaultdict(lambda: defaultdict(list))
        parser = argparse.ArgumentParser()
        parser.add_argument("--runindex", type=int, default=None)
        args = parser.parse_args(sys.argv[1:])
        for i in range(gc.total_num_runs):
            print("RUN {} of {}".format(i+1, gc.total_num_runs), flush=True)
            if args.runindex is not None and args.runindex != i:
                print("Skipping due to --runindex\n")
                continue
            gc.seed = gc.start_seed+i
            gc.rand_state = np.random.RandomState(seed=gc.seed)
            print_settings()
            for runner_id, runner in self._single_runners.items():
                if gc.mode != "test":
                    runner.train()
                if gc.mode != "train":
                    result = runner.test()
                    filename = "results/{}_{}.pkl".format(runner_id, i+1)
                    with open(filename, "wb") as f:
                        pickle.dump(result, f)
                    print("Dumped result to {}.".format(filename), flush=True)
                    for env_name, res in result.items():
                        all_results[runner_id][env_name].append(res)
                # Comment out above, uncomment below to load results from cache.
                # filename = "saved_results/{}_{}.pkl".format(runner_id, i+1)
                # with open(filename, "rb") as f:
                #     result = pickle.load(f)
                # # if "Stilman" in runner_id:
                # #     runner_id = "tampnamo_TAMPSolver_Stilman_10"
                # for env_name, res in result.items():
                #     all_results[runner_id][env_name].append(res)
            return_data = defaultdict(dict)
            plancost_data = defaultdict(dict)
            obj_value_data = defaultdict(dict)
            for runner_id, env_name_to_res in all_results.items():
                print("Runner ID: {}".format(runner_id))
                for env_name, res in env_name_to_res.items():
                    print("\tTest environment: {}".format(env_name))
                    print("\t\tNum runs done: {}".format(len(res)))
                    mean_res = np.mean(res, axis=0)
                    std_res = np.std(res, axis=0)
                    print("\t\tAverage results: {}".format(mean_res))
                    print("\t\tSD results: {}".format(std_res))
                    return_data[env_name][runner_id.split("_")[2]] = (
                        mean_res[0], std_res[0])
                    plancost_data[env_name][runner_id.split("_")[2]] = (
                        mean_res[1], std_res[1])
                    obj_value_data[env_name][runner_id.split("_")[2]] = (
                        mean_res[2], std_res[2])
            if args.runindex is None:
                self._plot(return_data,
                           "Avg Returns ({} runs)".format(i+1),
                           "returns")
                self._plot(plancost_data,
                           "Avg Plan Cost ({} runs)".format(i+1),
                           "solvecosts")
                self._plot(obj_value_data,
                           "Avg Objective ({} runs)".format(i+1),
                           "objvalues")

    @staticmethod
    def _plot(data, title, namestr):
        if "enemies" in ec.family_to_run:
            # Group plots by easy/medium/hard.
            Runner._plot_grouped_enemies(data, title, namestr)
            return
        if ec.family_to_run == "tamppickplace":
            # Put all plots together.
            Runner._plot_grouped_tamppickplace(data, title, namestr)
            return
        if ec.family_to_run == "tampnamo":
            # Put all plots together.
            Runner._plot_grouped_tampnamo(data, title, namestr)
            return
        if ec.family_to_run == "tampbins":
            # Put all plots together.
            Runner._plot_grouped_tampbins(data, title, namestr)
            return
        if ec.family_to_run == "dinner":
            # Put all plots together.
            Runner._plot_grouped_dinner(data, title, namestr)
            return
        for env_name, approach_name_to_res in data.items():
            names = []
            vals = []
            errors = []
            for approach_name, (mean, std) in approach_name_to_res.items():
                names.append(APPROACH_TO_NAME[approach_name])
                vals.append(mean)
                errors.append(std)
            plt.figure()
            plt.errorbar(names, vals, errors, fmt="o")
            plt.title("{}: {} {}".format(title, env_name, sc.solver_name))
            plt.xticks(rotation=45)
            plt.tight_layout()
            fname = "plots/{}_{}_{}.png".format(
                env_name, sc.solver_name, namestr)
            plt.savefig(fname, dpi=300)
            plt.close()
            print("Plot written out to {}".format(fname))

    @staticmethod
    def _plot_2d(return_data, plancost_data):
        ret_means = defaultdict(list)
        ret_stds = defaultdict(list)
        for env_name, approach_name_to_res in return_data.items():
            if ec.family_to_run.startswith("enemies") and "Medium" not in env_name:
                continue
            for approach_name, (mean, std) in approach_name_to_res.items():
                ret_means[approach_name].append(mean)
                ret_stds[approach_name].append(std)
        cost_means = defaultdict(list)
        cost_stds = defaultdict(list)
        for env_name, approach_name_to_res in plancost_data.items():
            if ec.family_to_run.startswith("enemies") and "Medium" not in env_name:
                continue
            for approach_name, (mean, std) in approach_name_to_res.items():
                if ec.family_to_run == "enemiesbig":
                    mean = 0.25  # timeout
                    std = 0
                cost_means[approach_name].append(mean)
                cost_stds[approach_name].append(std)
        if ec.family_to_run == "enemiesbig":
            plt.figure(figsize=[2.9, 4.8])
        else:
            plt.figure(figsize=[6.4, 4.8])
        for approach_name in ret_means:
            name = APPROACH_TO_NAME[approach_name]
            ret_val = np.mean(ret_means[approach_name])
            ret_error = np.mean(ret_stds[approach_name])
            cost_val = np.mean(cost_means[approach_name])
            cost_error = np.mean(cost_stds[approach_name])
            plt.errorbar(cost_val, ret_val, xerr=cost_error, yerr=ret_error,
                         fmt="o", label=name, markersize=20)
        # plt.errorbar(0,0,xerr=0,yerr=0,fmt="o",label="Stilman (NAMO only)")
        plt.xlabel("Computational Cost (seconds)")
        plt.ylabel("Returns")
        plt.title(ENV_TO_NAME[ec.family_to_run])
        fname = "plots/2d_{}_{}.png".format(ec.family_to_run, sc.solver_name)
        if ec.family_to_run != "enemiesbig":
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                       ncol=2, fancybox=True, shadow=True)
            # plt.legend(ncol=7)
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Plot written out to {}".format(fname))

    @staticmethod
    def _plot_grouped_enemies(data, title, namestr):
        easy_means = defaultdict(list)
        medium_means = defaultdict(list)
        hard_means = defaultdict(list)
        easy_stds = defaultdict(list)
        medium_stds = defaultdict(list)
        hard_stds = defaultdict(list)
        for env_name, approach_name_to_res in data.items():
            for approach_name, (mean, std) in approach_name_to_res.items():
                if "Easy" in env_name:
                    easy_means[approach_name].append(mean)
                    easy_stds[approach_name].append(std)
                elif "Medium" in env_name:
                    medium_means[approach_name].append(mean)
                    medium_stds[approach_name].append(std)
                elif "Hard" in env_name:
                    hard_means[approach_name].append(mean)
                    hard_stds[approach_name].append(std)
        names = []
        vals = []
        errors = []
        for approach_name in easy_means:
            names.append(APPROACH_TO_NAME[approach_name])
            vals.append(np.mean(easy_means[approach_name]))
            errors.append(np.mean(easy_stds[approach_name]))
        plt.figure()
        plt.errorbar(names, vals, errors, fmt="o")
        plt.title("{}: {} {}, {}".format(title, ENV_TO_NAME[ec.family_to_run], "Easy", sc.solver_name))
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = "plots/{}_{}_{}.png".format(
            "easy", sc.solver_name, namestr)
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Plot written out to {}".format(fname))
        names = []
        vals = []
        errors = []
        for approach_name in medium_means:
            names.append(APPROACH_TO_NAME[approach_name])
            vals.append(np.mean(medium_means[approach_name]))
            errors.append(np.mean(medium_stds[approach_name]))
        plt.figure()
        plt.errorbar(names, vals, errors, fmt="o")
        plt.title("{}: {} {}, {}".format(title, ENV_TO_NAME[ec.family_to_run], "Medium", sc.solver_name))
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = "plots/{}_{}_{}.png".format(
            "medium", sc.solver_name, namestr)
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Plot written out to {}".format(fname))
        names = []
        vals = []
        errors = []
        for approach_name in hard_means:
            names.append(APPROACH_TO_NAME[approach_name])
            vals.append(np.mean(hard_means[approach_name]))
            errors.append(np.mean(hard_stds[approach_name]))
        plt.figure()
        plt.errorbar(names, vals, errors, fmt="o")
        plt.title("{}: {} {}, {}".format(title, ENV_TO_NAME[ec.family_to_run], "Hard", sc.solver_name))
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = "plots/{}_{}_{}.png".format(
            "hard", sc.solver_name, namestr)
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Plot written out to {}".format(fname))

    @staticmethod
    def _plot_grouped_tamppickplace(data, title, namestr):
        mode = "together"
        # mode = "grouped"
        if mode == "together":
            means = defaultdict(list)
            stds = defaultdict(list)
            for env_name, approach_name_to_res in data.items():
                for approach_name, (mean, std) in approach_name_to_res.items():
                    means[approach_name].append(mean)
                    stds[approach_name].append(std)
            names = []
            vals = []
            errors = []
            for approach_name in means:
                names.append(APPROACH_TO_NAME[approach_name])
                vals.append(np.mean(means[approach_name]))
                errors.append(np.mean(stds[approach_name]))
            plt.figure()
            plt.errorbar(names, vals, errors, fmt="o")
            plt.title("{}: {}".format(title, ENV_TO_NAME[ec.family_to_run]))
            plt.xticks(rotation=45)
            plt.tight_layout()
            fname = "plots/TAMPPickPlace_{}_{}.png".format(sc.solver_name, namestr)
            plt.savefig(fname, dpi=300)
            plt.close()
            print("Plot written out to {}".format(fname))
        elif mode == "grouped":
            skinny_means = defaultdict(list)
            skinny_stds = defaultdict(list)
            fat_means = defaultdict(list)
            fat_stds = defaultdict(list)
            for env_name, approach_name_to_res in data.items():
                for approach_name, (mean, std) in approach_name_to_res.items():
                    if "Fat" in env_name:
                        fat_means[approach_name].append(mean)
                        fat_stds[approach_name].append(std)
                    else:
                        skinny_means[approach_name].append(mean)
                        skinny_stds[approach_name].append(std)
            names = []
            vals = []
            errors = []
            for approach_name in fat_means:
                names.append(APPROACH_TO_NAME[approach_name])
                vals.append(np.mean(fat_means[approach_name]))
                errors.append(np.mean(fat_stds[approach_name]))
            plt.figure()
            plt.errorbar(names, vals, errors, fmt="o")
            plt.title("{}: {} {}".format(title, "Fat", sc.solver_name))
            plt.xticks(rotation=45)
            plt.tight_layout()
            fname = "plots/TAMPPickPlace_{}_{}_{}.png".format("fat", sc.solver_name, namestr)
            plt.savefig(fname, dpi=300)
            plt.close()
            print("Plot written out to {}".format(fname))
            names = []
            vals = []
            errors = []
            for approach_name in skinny_means:
                names.append(APPROACH_TO_NAME[approach_name])
                vals.append(np.mean(skinny_means[approach_name]))
                errors.append(np.mean(skinny_stds[approach_name]))
            plt.figure()
            plt.errorbar(names, vals, errors, fmt="o")
            plt.title("{}: {} {}".format(title, "Skinny", sc.solver_name))
            plt.xticks(rotation=45)
            plt.tight_layout()
            fname = "plots/TAMPPickPlace_{}_{}_{}.png".format("skinny", sc.solver_name, namestr)
            plt.savefig(fname, dpi=300)
            plt.close()
            print("Plot written out to {}".format(fname))

    @staticmethod
    def _plot_grouped_tampnamo(data, title, namestr):
        means = defaultdict(list)
        stds = defaultdict(list)
        for env_name, approach_name_to_res in data.items():
            for approach_name, (mean, std) in approach_name_to_res.items():
                means[approach_name].append(mean)
                stds[approach_name].append(std)
        names = []
        vals = []
        errors = []
        for approach_name in means:
            names.append(APPROACH_TO_NAME[approach_name])
            vals.append(np.mean(means[approach_name]))
            errors.append(np.mean(stds[approach_name]))
        plt.figure()
        plt.errorbar(names, vals, errors, fmt="o")
        plt.title("{}: {}".format(title, ENV_TO_NAME[ec.family_to_run]))
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = "plots/TAMPNAMO_{}_{}.png".format(sc.solver_name, namestr)
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Plot written out to {}".format(fname))

    @staticmethod
    def _plot_grouped_tampbins(data, title, namestr):
        means = defaultdict(list)
        stds = defaultdict(list)
        for env_name, approach_name_to_res in data.items():
            for approach_name, (mean, std) in approach_name_to_res.items():
                means[approach_name].append(mean)
                stds[approach_name].append(std)
        names = []
        vals = []
        errors = []
        for approach_name in means:
            names.append(APPROACH_TO_NAME[approach_name])
            vals.append(np.mean(means[approach_name]))
            errors.append(np.mean(stds[approach_name]))
        plt.figure()
        plt.errorbar(names, vals, errors, fmt="o")
        plt.title("{}: {}".format(title, ENV_TO_NAME[ec.family_to_run]))
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = "plots/TAMPBins_{}_{}.png".format(sc.solver_name, namestr)
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Plot written out to {}".format(fname))

    @staticmethod
    def _plot_grouped_dinner(data, title, namestr):
        means = defaultdict(list)
        stds = defaultdict(list)
        for env_name, approach_name_to_res in data.items():
            for approach_name, (mean, std) in approach_name_to_res.items():
                means[approach_name].append(mean)
                stds[approach_name].append(std)
        names = []
        vals = []
        errors = []
        for approach_name in means:
            names.append(APPROACH_TO_NAME[approach_name])
            vals.append(np.mean(means[approach_name]))
            errors.append(np.mean(stds[approach_name]))
        plt.figure()
        plt.errorbar(names, vals, errors, fmt="o")
        plt.title("{}: {}".format(title, ENV_TO_NAME[ec.family_to_run]))
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = "plots/Dinner_{}_{}.png".format(sc.solver_name, namestr)
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Plot written out to {}".format(fname))

    @staticmethod
    def _create_runner_id(approach_name):
        return "{}_{}_{}_{}".format(ec.family_to_run, sc.solver_name,
                                    approach_name, ac.lam)


class SingleRunner:
    """Main runner class for a single approach and train/test split.
    """
    def __init__(self, approach_name, solver, train_env_names, test_env_names):
        self._approach_name = approach_name
        self._approach = getattr(approaches, approach_name)(solver)
        if gc.mode != "test":
            self._train_envs = [getattr(envs, env_name)()
                                for env_name in train_env_names]
        if gc.mode != "train":
            self._test_envs = [getattr(envs, env_name)()
                               for env_name in test_env_names]

    def train(self):
        """Main method for training a single approach.
        """
        print("\nTRAINING APPROACH {}...\n".format(self._approach_name))
        start = time.time()
        self._approach.train(self._train_envs)
        print("Training approach {} took {:.3f} sec\n\n".format(
            self._approach_name, time.time()-start))

    def test(self):
        """Main method for testing a single approach.
        """
        print("\nTESTING APPROACH {}...\n".format(self._approach_name))
        all_results = {}
        start = time.time()
        for env in self._test_envs:
            env_name = env.__class__.__name__
            print("Testing env: {}".format(env_name))
            results = utils.test_approach(
                env, self._approach, render=gc.do_render, verbose=True)
            all_results[env_name] = results
        print("Testing approach {} took {:.3f} sec\n\n".format(
            self._approach_name, time.time()-start))
        return all_results


def _main():
    Runner().run()
    # Runner.plot()


if __name__ == "__main__":
    _main()
