"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import os
import time as time
import matplotlib.pyplot as plt
import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from gym.wrappers.record_video import RecordVideo


env = gym.make("Taxi-v3", render_mode="rgb_array")
env_human = gym.make("Taxi-v3", render_mode="human")
n_actions = env.action_space.n  # type: ignore


#################################################
# 1. Play with QLearningAgent
#################################################
def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()
    for _ in range(t_max):
        # Get agent to pick action given state s

        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        total_reward += r
        if done:
            break
        agent.update(s, a, r, next_s)
        s = next_s
        # END SOLUTION

    return total_reward


def train(env, agent) -> tuple[np.array, QLearningAgent | SarsaAgent]:
    rewards = []
    times = []
    start_time = time.time()
    for _ in range(1000):
        rewards.append(play_and_train(env, agent))
        times.append(time.time() - start_time)
    print("mean reward", np.mean(rewards[-100:]))
    res = np.array([rewards, times]).reshape(2, -1)
    return res, agent


def plot_result(
    result: np.array, legend="Legend", plot_hline=False, plot_vline=False
) -> None:
    rewards, times = result[0], result[1]
    ema_rewards = np.convolve(rewards, np.ones(100), "valid") / 100
    ema_times = times[-len(ema_rewards) :]
    plt.plot(ema_times, ema_rewards, label=legend)
    plt.xlabel("time (seconds)")
    plt.ylabel("Moving average of rewards (window=100)")

    if plot_hline:
        last_reward = ema_rewards[-1]
        # Add a horizontal line at the last reward value
        plt.axhline(y=last_reward, linestyle="--", color="gray")
        plt.text(
            ema_times[0], last_reward, f"{last_reward:.2f}", verticalalignment="bottom"
        )

    if plot_vline:
        last_time = ema_times[-1]
        # Add a vertical line at the last time value
        plt.axvline(x=last_time, linestyle="--", color="gray")
        plt.text(
            last_time,
            np.min(ema_rewards),
            f"{last_time:.2f}",
            verticalalignment="bottom",
        )


def evaluate_hyperparameter(
    env, n_actions, hyperparameters, default_params, agent_class
):
    best_params = default_params.copy()

    for param_name, param_values in hyperparameters.items():
        times = []
        results = []
        values = []

        for value in param_values:
            params = best_params.copy()
            params[param_name] = value

            agent = agent_class(**{**params, "legal_actions": list(range(n_actions))})
            result, agent = train(env, agent)
            times.append(result[1][-1])
            results.append(result)
            values.append(value)

        best_idx = np.argmin(times)
        best_value = param_values[best_idx]
        best_params[param_name] = best_value
        other_params = {p: v for p, v in best_params.items() if p != param_name}
        plt.figure()
        for idx, (result, value) in enumerate(zip(results, values)):
            plot_hline = idx == best_idx
            plot_vline = idx == best_idx
            best_time = result[1][-1]
            best_reward = np.mean(result[0][-100:])
            plot_result(
                result,
                f"{param_name}={value}",
                plot_hline=plot_hline,
                plot_vline=plot_vline,
            )
        plt.title(
            f"Effect of {param_name} on {agent_class.__name__}\nParams: {other_params}, Best {param_name}: {best_value}"
        )
        plt.legend()
        directory = f"artefacts/{agent_class.__name__}/{param_name}"
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(
            directory, f"{str(best_params)}, R={best_reward}, T={best_time}.jpeg"
        )
        plt.savefig(file_path, format="jpeg")
        plt.show()

    return best_params, agent



default_params = {"learning_rate": 0.5, "epsilon": 0.5, "gamma": 0.99}
hyperparameters = {
    "learning_rate": [0.25, 0.5, 0.9, 0.99],
    "epsilon": [0.01, 0.1, 0.5, 0.9],
    "gamma": [0.1, 0.5, 0.9, 0.99, 1],
}

best_params, agent = evaluate_hyperparameter(
    env, n_actions, hyperparameters, default_params, QLearningAgent
)
print("Best params for QLearningAgent", best_params)

env_ql_r = RecordVideo(env = env, video_folder="artefacts/QLearningAgent", episode_trigger=lambda ep: True)
play_and_train(env_ql_r, agent, t_max=200)
env_ql_r.close()


#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


default_params = {
    "learning_rate": best_params["learning_rate"],
    "epsilon_start": 1,
    "epsilon_end": best_params["epsilon"],
    "gamma": best_params["gamma"],
}
hyperparameters = {
    "epsilon_start": [0.99, 0.75, 0.25],
}

best_params, agent = evaluate_hyperparameter(
    env, n_actions, hyperparameters, default_params, QLearningAgentEpsScheduling
)

print("Best params for QLearningAgentEpsScheduling", best_params)
env_qles_r = RecordVideo(env = env, video_folder="artefacts/QLearningAgentEpsScheduling", episode_trigger=lambda ep: True)
play_and_train(env_qles_r, agent, t_max=200)
env_qles_r.close()

####################
# 3. Play with SARSA
####################

default_params = {"learning_rate": 0.5, "gamma": 0.99}
hyperparameters = {
    "learning_rate": [0.25, 0.5, 0.9, 0.99],
    "gamma": [0.9, 0.95, 0.99, 1],
}

best_params, agent = evaluate_hyperparameter(
    env, n_actions, hyperparameters, default_params, SarsaAgent
)
print("Best params for SarsaAgent", best_params)
env_sarsa_r = RecordVideo(env = env, video_folder="artefacts/SarsaAgent", episode_trigger=lambda ep: True)
play_and_train(env_sarsa_r, agent, t_max=200)
env_sarsa_r.close()
