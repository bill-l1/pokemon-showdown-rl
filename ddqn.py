import asyncio

from poke_env.player import (
    RandomPlayer,
    wrap_for_old_gym_api,
)

import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers.legacy import Adam

from rlplayer import RLPlayer, evaluate_model
from config import BATTLE_FORMAT, SAVE_WEIGHTS, NUM_STEPS

async def main():
    opponent = RandomPlayer(battle_format=BATTLE_FORMAT)
    train_env = RLPlayer(battle_format=BATTLE_FORMAT, opponent=opponent, start_challenging=True)
    train_env = wrap_for_old_gym_api(train_env)

    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

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
        nb_steps=NUM_STEPS,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )

    tf.compat.v1.experimental.output_all_intermediates(True)
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    callbacks = None
    if SAVE_WEIGHTS:
        results_path = './results/'
        weights_filename = results_path + f"dqn_{BATTLE_FORMAT}_weights.h5f"
        checkpoint_weights_filename = results_path + f'dqn_{BATTLE_FORMAT}' + '_weights_{step}.h5f'
        log_filename = results_path + f'dqn_{BATTLE_FORMAT}_log.json'
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000)]
        callbacks += [FileLogger(log_filename, interval=10)]
    
    dqn.fit(train_env, callbacks=callbacks, nb_steps=NUM_STEPS, verbose=1)
    print(f"Training done with #{train_env.n_finished_battles} battles against random player")

    # dqn.load_weights(f'./results/dqn_{BATTLE_FORMAT}_weights.h5f')

    if SAVE_WEIGHTS:
        dqn.save_weights(weights_filename, overwrite=True)

    train_env.close()

    evaluate_model(dqn)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())


