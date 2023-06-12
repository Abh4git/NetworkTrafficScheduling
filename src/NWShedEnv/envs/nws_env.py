import bisect
import datetime
import random

import pandas as pd
import gym
import numpy as np
import plotly.figure_factory as ff
from pathlib import Path


class NwsEnv(gym.Env):
    def __init__(self, env_config=None):
        """
        This environment model the network scheduling problem as a single agent problem:

        -The actions correspond to a flow allocation + one action for no allocation at this time step (NOPE action)

        -We keep a time with next possible time steps

        -Each time we allocate a flow fragment, the end of the flow is added to the stack of time steps

        -If we don't have a legal action (i.e. we can't allocate a flow), - possibly deadline not met
        we automatically go to the next time step until we have a legal action

        -
        :param env_config: Ray dictionary of config parameter
        """
        if env_config is None:
            env_config = {
                "instance_path": str(Path(__file__).parent.absolute())
                + "/instances/ta80"
            }
        instance_path = env_config["instance_path"]

        # initial values for variables used for instance
        self.flows = 0
        self.links = 0
        self.instance_matrix = None
        self.jobs_length = None
        self.max_time_op = 0
        self.max_time_flows = 0
        self.nb_legal_actions = 0
        self.nb_link_legal = 0
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_flows = list()
        self.legal_actions = None
        self.time_until_available_link = None
        self.time_until_finish_current_op_flows = None
        self.todo_time_step_flow = None
        self.total_perform_op_time_flows = None
        self.needed_link_flows = None
        self.total_idle_time_flows = None
        self.idle_time_flows_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.link_legal = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0
        instance_file = open(instance_path, "r")
        line_str = instance_file.readline()
        line_cnt = 1
        while line_str:
            split_data = line_str.split()
            if line_cnt == 1:
                self.flows, self.links = int(split_data[0]), int(split_data[1])
                # matrix which store tuple of (machine, length of the job)
                self.instance_matrix = np.zeros(
                    (self.flows, self.links), dtype=(int, 2)
                )
                # contains all the time to complete flows
                self.flows_length = np.zeros(self.flows, dtype=int)
            else:
                # couple (machine, time)
                assert len(split_data) % 2 == 0
                # each jobs must pass a number of operation equal to the number of machines
                assert len(split_data) / 2 == self.links
                i = 0
                # we get the actual jobs
                flow_nb = line_cnt - 2
                while i < len(split_data):
                    link, time = int(split_data[i]), int(split_data[i + 1])
                    self.instance_matrix[flow_nb][i // 2] = (link, time)
                    self.max_time_op = max(self.max_time_op, time)
                    self.flows_length[flow_nb] += time
                    self.sum_op += time
                    i += 2
            line_str = instance_file.readline()
            line_cnt += 1
        instance_file.close()
        self.max_time_flows = max(self.flows_length)
        # check the parsed data are correct
        assert self.max_time_op > 0
        assert self.max_time_flows > 0
        assert self.flows > 0
        assert self.links > 1, "We need at least 2 machines"
        assert self.instance_matrix is not None
        # allocate a job + one to wait
        self.action_space = gym.spaces.Discrete(self.flows + 1)
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.links)
        ]
        """
        matrix with the following attributes for each flow:
            -Legal flow
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next link available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        """
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.flows + 1,)),
                "real_obs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.flows, 7), dtype=float
                ),
            }
        )

    def _get_current_state_representation(self):
        self.state[:, 0] = self.legal_actions[:-1]
        return {
            "real_obs": self.state,
            "action_mask": self.legal_actions,
        }

    def get_legal_actions(self):
        return self.legal_actions

    def reset(self):
        self.current_time_step = 0
        self.next_time_step = list()
        self.next_flows = list()
        self.nb_legal_actions = self.flows
        self.nb_link_legal = 0
        # represent all the legal actions
        self.legal_actions = np.ones(self.flows + 1, dtype=bool)
        self.legal_actions[self.flows] = False
        # used to represent the solution
        self.solution = np.full((self.flows, self.links), -1, dtype=int)
        self.time_until_available_link = np.zeros(self.links, dtype=int)
        self.time_until_finish_current_op_flows = np.zeros(self.flows, dtype=int)
        self.todo_time_step_flow = np.zeros(self.flows, dtype=int)
        self.total_perform_op_time_flows = np.zeros(self.flows, dtype=int)
        self.needed_link_flows= np.zeros(self.flows, dtype=int)
        self.total_idle_time_flows = np.zeros(self.flows, dtype=int)
        self.idle_time_flows_last_op = np.zeros(self.flows, dtype=int)
        self.illegal_actions = np.zeros((self.links, self.flows), dtype=bool)
        self.action_illegal_no_op = np.zeros(self.flows, dtype=bool)
        self.link_legal = np.zeros(self.links, dtype=bool)
        for flow in range(self.flows):
            needed_link = self.instance_matrix[flow][0][0]
            self.needed_link_flows[flow] = needed_link
            if not self.link_legal[needed_link]:
                self.link_legal[needed_link] = True
                self.nb_link_legal += 1
        self.state = np.zeros((self.flows, 7), dtype=float)
        return self._get_current_state_representation()

    def _prioritization_non_final(self):
        if self.nb_link_legal >= 1:
            for link in range(self.links):
                if self.link_legal[link]:
                    final_flow = list()
                    non_final_flow = list()
                    min_non_final = float("inf")
                    for flow in range(self.flows):
                        if (
                            self.needed_link_flows[flow] == link
                            and self.legal_actions[flow]
                        ):
                            if self.todo_time_step_flow[flow] == (self.links - 1):
                                final_flow.append(flow)
                            else:
                                current_time_step_non_final = self.todo_time_step_flow[
                                    flow
                                ]
                                time_needed_legal = self.instance_matrix[flow][
                                    current_time_step_non_final
                                ][1]
                                link_needed_nextstep = self.instance_matrix[flow][
                                    current_time_step_non_final + 1
                                ][0]
                                if (
                                    self.time_until_available_link[
                                        link_needed_nextstep
                                    ]
                                    == 0
                                ):
                                    min_non_final = min(
                                        min_non_final, time_needed_legal
                                    )
                                    non_final_flow.append(flow)
                    if len(non_final_flow) > 0:
                        for flow in final_flow:
                            current_time_step_final = self.todo_time_step_flow[flow]
                            time_needed_legal = self.instance_matrix[flow][
                                current_time_step_final
                            ][1]
                            if time_needed_legal > min_non_final:
                                self.legal_actions[flow] = False
                                self.nb_legal_actions -= 1

    def _check_no_op(self):
        self.legal_actions[self.flows] = False
        if (
            len(self.next_time_step) > 0
            and self.nb_link_legal <= 3
            and self.nb_legal_actions <= 4
        ):
            link_next = set()
            next_time_step = self.next_time_step[0]
            max_horizon = self.current_time_step
            max_horizon_machine = [
                self.current_time_step + self.max_time_op for _ in range(self.links)
            ]
            for flow in range(self.flows):
                if self.legal_actions[flow]:
                    time_step = self.todo_time_step_flow[flow]
                    link_needed = self.instance_matrix[flow][time_step][0]
                    time_needed = self.instance_matrix[flow][time_step][1]
                    end_flow = self.current_time_step + time_needed
                    if end_flow < next_time_step:
                        return
                    max_horizon_machine[link_needed] = min(
                        max_horizon_machine[link_needed], end_flow
                    )
                    max_horizon = max(max_horizon, max_horizon_machine[link_needed])
            for flow in range(self.flows):
                if not self.legal_actions[flow]:
                    if (
                        self.time_until_finish_current_op_flows[flow] > 0
                        and self.todo_time_step_flow[flow] + 1 < self.links
                    ):
                        time_step = self.todo_time_step_flow[flow] + 1
                        time_needed = (
                            self.current_time_step
                            + self.time_until_finish_current_op_flows[flow]
                        )
                        while (
                            time_step < self.links - 1 and max_horizon > time_needed
                        ):
                            link_needed = self.instance_matrix[flow][time_step][0]
                            if (
                                max_horizon_machine[link_needed] > time_needed
                                and self.link_legal[link_needed]
                            ):
                                link_next.add(link_needed)
                                if len(link_next) == self.nb_link_legal:
                                    self.legal_actions[self.flows] = True
                                    return
                            time_needed += self.instance_matrix[flow][time_step][1]
                            time_step += 1
                    elif (
                        not self.action_illegal_no_op[flow]
                        and self.todo_time_step_flow[flow] < self.links
                    ):
                        time_step = self.todo_time_step_flow[flow]
                        link_needed = self.instance_matrix[flow][time_step][0]
                        time_needed = (
                            self.current_time_step
                            + self.time_until_available_link[link_needed]
                        )
                        while (
                            time_step < self.links - 1 and max_horizon > time_needed
                        ):
                            link_needed = self.instance_matrix[flow][time_step][0]
                            if (
                                max_horizon_machine[link_needed] > time_needed
                                and self.link_legal[link_needed]
                            ):
                                link_next.add(link_needed)
                                if len(link_next) == self.nb_link_legal:
                                    self.legal_actions[self.flows] = True
                                    return
                            time_needed += self.instance_matrix[flow][time_step][1]
                            time_step += 1

    def step(self, action: int):
        reward = 0.0
        if action == self.flows:
            self.nb_link_legal = 0
            self.nb_legal_actions = 0
            for flow in range(self.flows):
                if self.legal_actions[flow]:
                    self.legal_actions[flow] = False
                    needed_link = self.needed_link_flows[flow]
                    self.link_legal[needed_link] = False
                    self.illegal_actions[needed_link][flow] = True
                    self.action_illegal_no_op[flow] = True
            while self.nb_link_legal == 0:
                reward -= self.increase_time_step()
            scaled_reward = self._reward_scaler(reward)
            self._prioritization_non_final()
            self._check_no_op()
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                {},
            )
        else:
            current_time_step_flow = self.todo_time_step_flow[action]
            link_needed = self.needed_link_flows[action]
            time_needed = self.instance_matrix[action][current_time_step_flow][1]
            reward += time_needed
            self.time_until_available_link[link_needed] = time_needed
            self.time_until_finish_current_op_flows[action] = time_needed
            self.state[action][1] = time_needed / self.max_time_op
            to_add_time_step = self.current_time_step + time_needed
            if to_add_time_step not in self.next_time_step:
                index = bisect.bisect_left(self.next_time_step, to_add_time_step)
                self.next_time_step.insert(index, to_add_time_step)
                self.next_flows.insert(index, action)
            self.solution[action][current_time_step_flow] = self.current_time_step
            for flow in range(self.flows):
                if (
                    self.needed_link_flows[flow] == link_needed
                    and self.legal_actions[flow]
                ):
                    self.legal_actions[flow] = False
                    self.nb_legal_actions -= 1
            self.nb_link_legal -= 1
            self.link_legal[link_needed] = False
            for flow in range(self.flows):
                if self.illegal_actions[link_needed][flow]:
                    self.action_illegal_no_op[flow] = False
                    self.illegal_actions[link_needed][flow] = False
            # if we can't allocate new job in the current timestep, we pass to the next one
            while self.nb_link_legal == 0 and len(self.next_time_step) > 0:
                reward -= self.increase_time_step()
            self._prioritization_non_final()
            self._check_no_op()
            # we then need to scale the reward
            scaled_reward = self._reward_scaler(reward)
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                {},
            )

    def _reward_scaler(self, reward):
        return reward / self.max_time_op

    def increase_time_step(self):
        """
        The heart of the logic his here, we need to increase every counter when we have a nope action called
        and return the time elapsed
        :return: time elapsed
        """
        hole_planning = 0
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.next_flows.pop(0)
        difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        for flow in range(self.flows):
            was_left_time = self.time_until_finish_current_op_flows[flow]
            if was_left_time > 0:
                performed_op_flow = min(difference, was_left_time)
                self.time_until_finish_current_op_flows[flow] = max(
                    0, self.time_until_finish_current_op_flows[flow] - difference
                )
                self.state[flow][1] = (
                    self.time_until_finish_current_op_flows[flow] / self.max_time_op
                )
                self.total_perform_op_time_flows[flow] += performed_op_flow
                self.state[flow][3] = (
                    self.total_perform_op_time_flows[flow] / self.max_time_flows
                )
                if self.time_until_finish_current_op_flows[flow] == 0:
                    self.total_idle_time_flows[flow] += difference - was_left_time
                    self.state[flow][6] = self.total_idle_time_flows[flow] / self.sum_op
                    self.idle_time_flows_last_op[flow] = difference - was_left_time
                    self.state[flow][5] = self.idle_time_flows_last_op[flow] / self.sum_op
                    self.todo_time_step_flow[flow] += 1
                    self.state[flow][2] = self.todo_time_step_flow[flow] / self.links
                    if self.todo_time_step_flow[flow] < self.links:
                        self.needed_link_flows[flow] = self.instance_matrix[flow][
                            self.todo_time_step_flow[flow]
                        ][0]
                        self.state[flow][4] = (
                            max(
                                0,
                                self.time_until_available_link[
                                    self.needed_link_flows[flow]
                                ]
                                - difference,
                            )
                            / self.max_time_op
                        )
                    else:
                        self.needed_link_flows[flow] = -1
                        # this allow to have 1 is job is over (not 0 because, 0 strongly indicate that the job is a
                        # good candidate)
                        self.state[flow][4] = 1.0
                        if self.legal_actions[flow]:
                            self.legal_actions[flow] = False
                            self.nb_legal_actions -= 1
            elif self.todo_time_step_flow[flow] < self.links:
                self.total_idle_time_flows[flow] += difference
                self.idle_time_flows_last_op[flow] += difference
                self.state[flow][5] = self.idle_time_flows_last_op[flow] / self.sum_op
                self.state[flow][6] = self.total_idle_time_flows[flow] / self.sum_op
        for link in range(self.links):
            if self.time_until_available_link[link] < difference:
                empty = difference - self.time_until_available_link[link]
                hole_planning += empty
            self.time_until_available_link[link] = max(
                0, self.time_until_available_link[link] - difference
            )
            if self.time_until_available_link[link] == 0:
                for flow in range(self.flows):
                    if (
                        self.needed_link_flows[flow] == link
                        and not self.legal_actions[flow]
                        and not self.illegal_actions[link][flow]
                    ):
                        self.legal_actions[flow] = True
                        self.nb_legal_actions += 1
                        if not self.link_legal[link]:
                            self.link_legal[link] = True
                            self.nb_link_legal += 1
        return hole_planning

    def _is_done(self):
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            self.last_solution = self.solution
            return True
        return False

    def render(self, mode="human"):
        df = []
        for flow in range(self.flows):
            i = 0
            while i < self.links and self.solution[flow][i] != -1:
                dict_op = dict()
                dict_op["Task"] = "Flow {}".format(flow)
                start_sec = self.start_timestamp + self.solution[flow][i]
                finish_sec = start_sec + self.instance_matrix[flow][i][1]
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = "Link {}".format(
                    self.instance_matrix[flow][i][0]
                )
                df.append(dict_op)
                i += 1
        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(
                df,
                index_col="Resource",
                colors=self.colors,
                show_colorbar=True,
                group_tasks=True,
            )
            fig.update_yaxes(
                autorange="reversed"
            )  # otherwise tasks are listed from the bottom up
        return fig
