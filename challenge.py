import asyncio

import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers.legacy import Adam

from rlplayer import RLPlayer, TrainedPlayer
from config import BATTLE_FORMAT, NUM_STEPS

async def main():
    train_env = RLPlayer(battle_format=BATTLE_FORMAT, opponent="placeholder", start_challenging=True)

    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape
    print(train_env.observation_space.shape)

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

    print("Loading weights...")
    dqn.load_weights(f'./results/dqn_{BATTLE_FORMAT}_weights.h5f')
    print("Weights loaded - now waiting for challenge")
    player = TrainedPlayer(model=dqn.model, rlplayer=train_env)
    username = ""
    await player.send_challenges(username, n_challenges=1)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())