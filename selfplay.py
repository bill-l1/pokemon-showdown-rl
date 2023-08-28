

import numpy as np
import asyncio
from threading import Thread

from poke_env.player import wrap_for_old_gym_api

import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers.legacy import Adam

from rlplayer import RLPlayer, evaluate_model
from config import BATTLE_FORMAT, SAVE_WEIGHTS, NUM_STEPS

GRAPH = tf.compat.v1.get_default_graph()

async def launch_battles(p1: RLPlayer, p2: RLPlayer, num_challenges):
    await asyncio.gather(
        p1.agent.accept_challenges(p2.username, num_challenges),
        p2.agent.send_challenges(p1.username, num_challenges)
    )

def train(player: RLPlayer, model: DQNAgent):
    with GRAPH.as_default():
        session = tf.compat.v1.keras.backend.get_session()
        init = tf.compat.v1.global_variables_initializer()
        session.run(init)
        model.fit(player, nb_steps=10000, verbose=True)
        player.done_training = True
        # Play out the remaining battles so both fit() functions complete
        while player.current_battle and not player.current_battle.finished:
            _ = player.step(np.random.choice(player.action_space.n))

if __name__ == "__main__":
    player1 = RLPlayer(
        battle_format=BATTLE_FORMAT,
        log_level=30,
        opponent="placeholder",
        start_challenging=False,
    )
    player1 = wrap_for_old_gym_api(player1)
    player2 = RLPlayer(
        battle_format=BATTLE_FORMAT,
        log_level=30,
        opponent="placeholder",
        start_challenging=False,
    )
    player2 = wrap_for_old_gym_api(player2)

    n_action = player1.action_space.n
    input_shape = (1,) + player1.observation_space.shape

    model = Sequential()
    model.add(Dense(256, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=500,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        # enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    player1.done_training = False
    player2.done_training = False

    loop = asyncio.get_event_loop()

    t1 = Thread(target=lambda: train(player1, dqn))
    t1.start()

    t2 = Thread(target=lambda: train(player2, dqn))
    t2.start()

    while not player1.done_training or not player2.done_training:
        loop.run_until_complete(launch_battles(player1, player2, 1))

    t1.join()
    t2.join()

    num_battles = player1.n_finished_battles
    print(f"Completed ${num_battles} battles of fictitious self-play")

    player1.close(purge=False)
    player2.close(purge=False)

    evaluate_model(dqn)