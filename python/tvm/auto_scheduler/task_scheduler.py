# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name

""" The task scheduler that allocates the time resources when tuning multiple tasks together

The details of the "gradient" strategy below can be found in the section 6 of this paper:
L. Zheng, C. Jia, M. Sun, Z. Wu, C. Yu, et al. "Ansor : Generating High-Performance Tensor
Programs for Deep Learning." (OSDI 2020).
"""

import time
import math
import logging

import numpy as np

from .search_policy import SearchPolicy, SketchPolicy
from .cost_model import RandomModel, XGBModel
from .utils import array_mean
from .measure import ProgramMeasurer
from .measure_record import RecordReader
from . import _ffi_api

logger = logging.getLogger("auto_scheduler")


def make_search_policies(
    search_policy, tasks, num_measures_per_round, verbose, load_model_file=None, load_log_file=None
):
    """Make a list of search policies for a list of search tasks.
    It creates one policy per task.

    Parameters
    ----------
    search_policy: Union[str, List[SearchPolicy]]
        The name of search policy.
    tasks: List[SearchTask]
        The list of all tasks
    num_measures_per_round: int
        The number of schedules to be measured at each search round.
        This should be the same as `TuningOptions.num_measures_per_round`
    verbose: int
        The verbosity level. 0 for silent.
    load_model_file: Optional[str]
        Load pre-trained model from this file. If this is None, the cost model will
        be trained from scratch.
    load_log_file: Optional[str]
        Load measurement records from this file. If it is not None, the status of the
        task scheduler, search policies and cost models will be restored according to this file.

    Returns
    -------
    policies: List[SearchPolicy]
        The list of search policies
    """
    if search_policy == "default":
        search_policy = "sketch.xgb"

    if isinstance(search_policy, str):
        policy_type, model_type = search_policy.split(".")
        if model_type == "xgb":
            cost_model = XGBModel(num_warmup_sample=len(tasks) * num_measures_per_round)
            if load_model_file:
                logger.info("TaskScheduler: Load pretrained model...")
                cost_model.load(load_model_file)
            elif load_log_file:
                cost_model.update_from_file(load_log_file)
        elif model_type == "random":
            cost_model = RandomModel()
        else:
            raise ValueError("Invalid search policy: " + search_policy)

        if policy_type == "sketch":
            search_policies = [SketchPolicy(task, cost_model, verbose=verbose) for task in tasks]
        else:
            raise ValueError("Invalid search policy: " + search_policy)
    else:
        # check type
        assert isinstance(search_policy, (tuple, list))
        for item in search_policy:
            assert isinstance(item, SearchPolicy)
        search_policies = search_policy

    return search_policies


def derive_similarity_tag(dag, log_base=1.618):
    """Derive the tag for similarity check from one computational DAG.
    The DAGs with the same tag are considered as similar tasks.

    The tag format is <op1-tag>_<op2-tag> ... <log(flop)>.

    If the tag is "", then the task is not considered to be similar to any other tasks.

    Parameters
    ----------
    dag: ComputeDAG
        The input computational DAG
    log_base: float = 1.618
        The base of log to normalize FLOPS

    Returns
    -------
    tag: str
        The tag of this computational DAG.
    """
    ret = ""
    for op in dag.ops:
        tag = op.attrs.get("auto_scheduler_task_scheduler_tag", None)
        if tag:
            ret += op.attrs["auto_scheduler_task_scheduler_tag"] + "_"
    if ret:
        ret += "%d" % int(math.log(dag.flop_ct + 1, log_base))
    return ret


class TaskScheduler:
    """
    Allocate the time resources when tuning multiple tasks together.
    This implements two strategies: "round-robin" and "gradient".

    Parameters
    ----------
    tasks: List[SearchTask]
        All tasks to tune
    task_weights: Optional[List[float]]
        The weights of tasks.
        If provided, the task scheduler will set the objective function to
        sum(weight[t] * latency[t]), where weight[t] is the weight of a task
        and the lantecy[t] is the lantecy of the task.
        If not provided, the task scheduer will assign equal weights to all
        tasks (i.e., the objective function is sum(latency[t])).
    objective_func: Optional[Callable[List[float] -> float]]
        The objective function to be minimized.
        The objective function accepts the current latencies of all tasks and returns the
        objective.
        If not provided, the objective is the weighted sum of the latencies of all tasks.
    strategy: str = "gradient"
        The scheduling strategy.
        "round-robin": Tune tasks in round robin order.
        "gradient" : Tune tasks with gradient descent.
    load_model_file: Optional[str]
        Load pre-trained model from this file. If this is None, the cost model will
        be trained from scratch.
    load_log_file: Optional[str]
        Load measurement records from this file. If it is not None, the status of the
        task scheduler, search policies and cost models will be restored according to this file.
    verbose: int = 1
        The level of verbosity. 0 means silent.
    alpha: float = 0.2
        The parameter used for 'gradient' strategy
    beta: float = 2
        The parameter used for 'gradient' strategy
    backward_window_size: int = 3
        The parameter used for 'gradient' strategy
    """

    def __init__(
        self,
        tasks,
        task_weights=None,
        objective_func=None,
        strategy="gradient",
        load_model_file: str = None,
        load_log_file: str = None,
        alpha: float = 0.2,
        beta: float = 2,
        gamma: float = 0.5,
        backward_window_size: int = 3,
    ):
        self.tasks = tasks
        if objective_func:  # use custom objective function
            self.objective_func = objective_func
        else:  # use weighted sum
            if task_weights:
                self.objective_func = lambda costs: sum(c * w for c, w in zip(costs, task_weights))
            else:
                self.objective_func = sum

        self.strategy = strategy
        self.load_log_file = load_log_file
        self.load_model_file = load_model_file
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.backward_window_size = backward_window_size

        assert len(self.tasks) != 0, "No tasks"
        assert self.strategy in ["round-robin", "gradient"]

        # task_cts[i] saves how many times task i is tuned
        self.task_cts = [0 for _ in range(len(self.tasks))]

        # task_costs_history[i] saves the latency history of task i
        self.task_costs_history = [[] for _ in range(len(self.tasks))]

        # best_costs[i] saves the best latency of task i
        self.best_costs = 1e10 * np.ones(len(self.tasks))
        self.cur_score = self._compute_score(self.best_costs)

        self.tune_option = self.measurer = self.search_policies = None
        self.ct = self.best_ct = self.best_score = self.tic = None
        self.num_measures_per_round = None
        self.dead_tasks = set()

        # Build similarity groups
        self.task_tags = []  # task_id -> tag
        self.tag_to_group_id = {}  # tag -> group_id
        self.group_task_ids = []  # group_id -> all task ids in this group
        self.flop_cts = []  # task_id -> the number of floating ops
        for i, task in enumerate(self.tasks):
            tag = derive_similarity_tag(task.compute_dag)
            self.task_tags.append(tag)
            self.flop_cts.append(task.compute_dag.flop_ct)
            if not tag:
                continue

            if tag not in self.tag_to_group_id:
                self.tag_to_group_id[tag] = len(self.tag_to_group_id)
                self.group_task_ids.append([])
            self.group_task_ids[self.tag_to_group_id[tag]].append(i)

    def tune(self, tune_option, search_policy="default"):
        """Tune a batch of tasks together.

        Parameters
        ----------
        tune_option: TuningOptions
            The options of tuning
        search_policy: : Union[str, List[SearchPolicy]]
            The list of search policies.
            If it is str.
            "sketch.xgb" for SketchPolicy + XGBModel
            "sketch.random" for SketchPolicy + RandomModel
        """
        # init members
        self.tune_option = tune_option
        early_stopping = 1e20 if tune_option.early_stopping < 0 else tune_option.early_stopping

        self.measurer = ProgramMeasurer(
            tune_option.builder,
            tune_option.runner,
            tune_option.measure_callbacks,
            tune_option.verbose,
        )
        self.ct = self.best_ct = 0
        self.tic = time.time()

        # reset num_measures_per_round to make sure every task is tuned at least once
        self.num_measures_per_round = min(
            tune_option.num_measures_per_round, tune_option.num_measure_trials // len(self.tasks)
        )
        if self.num_measures_per_round <= 0:
            raise ValueError("num_measure_trials is too small. Please set it to a higher value.")

        # restore the status of the task scheduler from a log file
        if self.load_log_file:
            self._restore_status(self.load_log_file, self.num_measures_per_round)

        # make one search policy for one task
        self.search_policies = make_search_policies(
            search_policy,
            self.tasks,
            self.num_measures_per_round,
            tune_option.verbose,
            self.load_model_file,
            self.load_log_file,
        )

        # do a round robin first to warm up
        for i in range(len(self.tasks)):
            self._tune_task(i)
        self.best_ct = self.ct
        self.best_score = self.cur_score

        # use the specific strategy to choose workload to tune
        task_idx = -1
        while self.ct < tune_option.num_measure_trials and len(self.dead_tasks) < len(self.tasks):
            if self.strategy == "round-robin":
                task_idx = (task_idx + 1) % len(self.tasks)
                while task_idx in self.dead_tasks:
                    task_idx = (task_idx + 1) % len(self.tasks)
            elif self.strategy == "gradient":
                gradients = []
                for i in range(len(self.tasks)):
                    if i in self.dead_tasks:
                        gradients.append(0)
                        continue

                    # compute gradient from chain rule : (delta f / delta g_i)
                    delta = 1e-4
                    new_costs = list(self.best_costs)
                    new_costs[i] -= delta
                    chain_grad = (
                        self._compute_score(self.best_costs) - self._compute_score(new_costs)
                    ) / delta

                    # compute (g_i(t_i) - g(t_i - \Delta t)) / (\Delta t)
                    if (
                        self.task_cts[i] - 1 < len(self.task_costs_history[i])
                        and self.task_cts[i] - 1 - self.backward_window_size >= 0
                    ):
                        backward_grad = (
                            self.task_costs_history[i][self.task_cts[i] - 1]
                            - self.task_costs_history[i][
                                self.task_cts[i] - 1 - self.backward_window_size
                            ]
                        ) / self.backward_window_size
                    else:
                        backward_grad = 0

                    # compute (g_i(t_i + \Delta t) - g(t_i)) / (\Delta t)
                    g_next_1 = self.best_costs[i] - (self.best_costs[i] / self.task_cts[i])

                    g_next_2 = self.beta * 1e30
                    group_id = self.tag_to_group_id.get(self.task_tags[i], None)
                    if group_id is not None and len(self.group_task_ids[group_id]) > 1:
                        best_flops = max(
                            [
                                self.flop_cts[j] / self.best_costs[j]
                                for j in self.group_task_ids[group_id]
                            ]
                        )
                        g_next_2 = self.beta * self.flop_cts[i] / best_flops

                    g_next = min(g_next_1, g_next_2)
                    forward_grad = g_next - self.best_costs[i]

                    # combine all grads
                    grad = chain_grad * (
                        self.alpha * backward_grad + (1 - self.alpha) * forward_grad
                    )
                    assert grad <= 0
                    gradients.append(grad)

                if max(gradients) == min(gradients):
                    task_idx = np.random.choice(len(gradients))
                else:
                    task_idx = np.argmin(gradients)
            else:
                raise ValueError("Invalid strategy: " + self.strategy)

            self._tune_task(task_idx)
            self._adjust_similarity_group(task_idx)

            if self.cur_score < self.best_score:
                self.best_score = self.cur_score
                self.best_ct = self.ct
            elif self.ct - self.best_ct >= early_stopping and all(
                cost < 1e9 for cost in self.best_costs
            ):
                if self.tune_option.verbose >= 1:
                    print(
                        "Stop early since no performance improvement in the last "
                        + str(early_stopping)
                        + " measurement trials."
                    )
                break

    def _print_table_info(self, next_task_idx):
        # table header
        _ffi_api.PrintTitle("Task Scheduler")
        print("|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |")
        print("-------------------------------------------------")

        # content
        for i in range(len(self.tasks)):
            id_str = "%d" % i
            latency_str = "%.3f" % (1e3 * self.best_costs[i]) if self.best_costs[i] < 1e9 else "-"
            speed_str = (
                "%.2f" % (self.tasks[i].compute_dag.flop_ct / self.best_costs[i] / 1e9)
                if self.best_costs[i] < 1e9
                else "-"
            )
            trials_str = "%d" % (self.task_cts[i] * self.num_measures_per_round)
            print("| %4s | %12s | % 14s | %6s |" % (id_str, latency_str, speed_str, trials_str))
        print("-------------------------------------------------")

        # overall info
        if all(cost < 1e9 for cost in self.best_costs):
            total_latency_str = "%.3f" % (self.cur_score * 1e3)
        else:
            total_latency_str = "-"
        print(
            "Estimated total latency: %s ms\tTrials: %d\tUsed time : %.0f s\tNext ID: %d\t"
            % (total_latency_str, self.ct, time.time() - self.tic, next_task_idx)
        )

    def _tune_task(self, task_idx):
        """Tune the select task for one round"""
        if self.tune_option.verbose >= 1:
            self._print_table_info(task_idx)

        measure_inputs, measure_results = self.search_policies[task_idx].continue_search_one_round(
            self.num_measures_per_round, self.measurer
        )

        for res in measure_results:
            cost = array_mean(res.costs)
            if cost < self.best_costs[task_idx]:
                self.best_costs[task_idx] = cost

        if len(measure_inputs) == 0:
            self.dead_tasks.add(task_idx)

        self.task_cts[task_idx] += 1
        self.task_costs_history[task_idx].append(self.best_costs[task_idx])

        self.ct += len(measure_inputs)
        self.cur_score = self._compute_score(self.best_costs)

    def _compute_score(self, costs):
        """compute the objective function"""
        return self.objective_func(costs)

    def _adjust_similarity_group(self, task_idx):
        """adjust the similarity group for the selected task"""
        group_id = self.tag_to_group_id.get(self.task_tags[task_idx], None)
        if group_id is None or len(self.group_task_ids[group_id]) <= 1:
            return

        group_ids = self.group_task_ids[group_id]
        best_group_flops = max([self.flop_cts[j] / self.best_costs[j] for j in group_ids])
        cur_flops = self.flop_cts[task_idx] / self.best_costs[task_idx]

        # if we tune a task for many times but it still cannot achieve
        # a similar speed to the fastest one in its group, this means this task
        # is actually not similar to other tasks in its group.
        # So we will remove it from its original group.
        if cur_flops < best_group_flops / self.beta and self.task_cts[task_idx] > 5 + max(
            self.task_cts[j] for j in group_ids if j != task_idx
        ):
            self.task_tags[task_idx] = None
            group_ids.remove(task_idx)

    def _restore_status(self, log_file, num_measures_per_round):
        """restore task_cts and best_costs from a log file"""
        str_target = str(self.tasks[0].target)
        workload_key_to_task_id = {t.workload_key: i for i, t in enumerate(self.tasks)}
        total_ct = -1

        for total_ct, (inp, res) in enumerate(RecordReader(log_file)):
            if str(inp.task.target) != str_target:
                continue
            task_idx = workload_key_to_task_id.get(inp.task.workload_key, None)
            if task_idx is None:
                continue

            if res.error_no == 0:
                self.best_costs[task_idx] = min(self.best_costs[task_idx], array_mean(res.costs))

            self.task_cts[task_idx] += 1

        for i in range(len(self.tasks)):
            # The computation of taks_cts is just an estimation.
            # The estimation may not be accurate if the log file is changed externally or
            # `num_measures_per_round` is different from the last tuning.
            self.task_cts[i] = int(self.task_cts[i] / num_measures_per_round + 0.5)
            self.task_costs_history[i].append(self.best_costs[i])

        logger.info("TaskScheduler: Loaded %d measurement records from %s", total_ct + 1, log_file)