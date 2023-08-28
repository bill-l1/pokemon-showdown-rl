import numpy as np

from poke_env.player import (
    Player,
    Gen9EnvSinglePlayer,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    wrap_for_old_gym_api,
)

from poke_env.environment.status import Status
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.effect import Effect
from poke_env.environment.weather import Weather
from poke_env.environment.field import Field
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.move_category import MoveCategory

from rl.agents.dqn import DQNAgent
import tensorflow as tf

from gym.spaces import Space, Box
from config import GEN_DATA, BATTLE_FORMAT

GRAPH = tf.compat.v1.get_default_graph()

def embed_species(pokemon: Pokemon):
        active_species = np.zeros(len(GEN_DATA.pokedex))
        active_species[GEN_DATA.pokedex[pokemon.species]['num']] = 1.
        return active_species
    
def embed_types(pokemon: Pokemon):
    active_types = np.zeros(len(PokemonType))
    active_types[pokemon.type_1.value-1] = 1.
    if pokemon.type_2:
        active_types[pokemon.type_2.value-1] = 1.
    return active_types

def embed_current_hp(pokemon: Pokemon):
    active_current_hp = np.array([pokemon.current_hp_fraction])
    return active_current_hp

def embed_fainted(pokemon: Pokemon):
    active_fainted = np.array([pokemon.fainted])
    return active_fainted

def embed_status(pokemon: Pokemon):
    active_status = np.zeros(len(Status))
    if pokemon.status:
        active_status[pokemon.status.value-1] = 1.
    return active_status

def embed_boosts(pokemon: Pokemon):
    active_boosts = np.zeros(7)
    for i, (_, val) in enumerate(pokemon.boosts.items()):
        # active_boosts[i] = val / 6.
        active_boosts[i] = val
    return active_boosts

def embed_effects(pokemon: Pokemon):
    active_effects = np.zeros(len(Effect))
    for effect, val in pokemon.effects.items():
        active_effects[effect.value-1] = 1.
    return active_effects

def embed_side_conditions(side_conditions: [SideCondition, int]):
    active_side_conditions = np.zeros(len(SideCondition))
    for sc, val in side_conditions.items():
        if sc.value == SideCondition.SPIKES:
            active_side_conditions[sc.value-1] = val / 3.
            # active_side_conditions[sc.value-1] = val
        elif sc.value == SideCondition.TOXIC_SPIKES:
            active_side_conditions[sc.value-1] = val / 2.
            # active_side_conditions[sc.value-1] = val
        else:
            active_side_conditions[sc.value-1] = 1.
    return active_side_conditions

def embed_weather(battle: AbstractBattle):
    weather = np.zeros(len(Weather))
    if len(battle.weather) > 0:
        weather[list(battle.weather)[0].value-1] = 1.
    return weather

def embed_fields(battle: AbstractBattle):
    fields = np.zeros(len(Field))
    for f, _ in battle.fields.items():
        fields[f.value-1] = 1.
    return fields

def embed_available_moves(battle: AbstractBattle):
    num_moves = len(GEN_DATA.moves)
    available_moves = np.zeros(num_moves * 4)
    for i, move in enumerate(battle.available_moves):
        if move.id in GEN_DATA.moves:
            available_moves[i * num_moves + GEN_DATA.moves[move.id]['num']] = 1.
    return available_moves

def _stat_estimation(mon, stat):
    # Stats boosts value
    if mon.boosts[stat] > 1:
        boost = (2 + mon.boosts[stat]) / 2
    else:
        boost = 2 / (2 - mon.boosts[stat])
    return ((2 * mon.base_stats[stat] + 31) + 5) * boost

def embed_available_move_damage(battle: AbstractBattle):
    move_damage = np.zeros(4)
    physical_ratio = _stat_estimation(battle.active_pokemon, "atk") / _stat_estimation(battle.opponent_active_pokemon, "def")
    special_ratio = _stat_estimation(battle.active_pokemon, "spa") / _stat_estimation(battle.opponent_active_pokemon, "spd")
    for i, move in enumerate(battle.available_moves):
        # TODO: factor in ability, weather, field and status
        move_damage[i] = move.base_power * (1.5 if move.type in battle.active_pokemon.types else 1) * (
                    physical_ratio if move.category == MoveCategory.PHYSICAL else special_ratio
                ) * move.expected_hits * battle.opponent_active_pokemon.damage_multiplier(move) / 100.
    return move_damage

def embed_available_switches(battle: AbstractBattle):
    num_mons = len(GEN_DATA.pokedex)
    available_switches = np.zeros(num_mons * 5)
    for i, pokemon in enumerate(battle.available_switches):
        available_switches[i * num_mons + GEN_DATA.pokedex[pokemon.species]['num']] = 1.
    return available_switches

def embed_battle(battle: AbstractBattle):
    # active mon species
    active_species = embed_species(battle.active_pokemon)
    # active mon types
    active_types = embed_types(battle.active_pokemon)
    # active mon current % hp
    active_current_hp = embed_current_hp(battle.active_pokemon)
    # active mon fainted
    active_fainted = embed_fainted(battle.active_pokemon)
    # active mon status
    active_status = embed_status(battle.active_pokemon)
    # active mon boosts
    active_boosts = embed_boosts(battle.active_pokemon)
    # active mon effects
    active_effects = embed_effects(battle.active_pokemon)

    # opponent's active mon species
    opp_active_species = embed_species(battle.opponent_active_pokemon)
    # active mon types
    opp_active_types = embed_types(battle.opponent_active_pokemon)
    # opponent's active mon current % hp
    opp_active_current_hp = embed_current_hp(battle.opponent_active_pokemon)
    # opponent's active mon fainted
    opp_active_fainted = embed_fainted(battle.opponent_active_pokemon)
    # opponent's active mon status
    opp_active_status = embed_status(battle.opponent_active_pokemon)
    # opponent's active mon boosts
    opp_active_boosts = embed_boosts(battle.opponent_active_pokemon)
    # opponent's active mon effects
    opp_active_effects = embed_effects(battle.opponent_active_pokemon)

    # side conditions
    side_conditions = embed_side_conditions(battle.side_conditions)
    opp_side_conditions = embed_side_conditions(battle.opponent_side_conditions)

    # weather
    weather = embed_weather(battle)
    # field
    fields = embed_fields(battle)
    # available moves
    available_moves = embed_available_moves(battle)
    # available moves' damage
    available_move_damage = embed_available_move_damage(battle)
    # available switches
    available_switches = embed_available_switches(battle)

    return np.concatenate(
        [
            active_species,
            active_types,
            active_current_hp,
            active_fainted,
            active_status,
            active_boosts,
            active_effects,
            opp_active_species,
            opp_active_types,
            opp_active_current_hp,
            opp_active_fainted,
            opp_active_status,
            opp_active_boosts,
            opp_active_effects,
            side_conditions,
            opp_side_conditions,
            weather,
            fields,
            available_moves,
            available_move_damage,
            available_switches
        ],
        dtype=np.float32
    )

class RLPlayer(Gen9EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(current_battle, fainted_value=2.0, hp_value=1.0, victory_value=10.0)
        # return self.reward_computing_helper(current_battle) # simple reward for win/loss: +1/-1
    
    def embed_battle(self, battle: AbstractBattle):
        return embed_battle(battle)
    
    def describe_low_embedding(self):
        active_species = np.zeros(len(GEN_DATA.pokedex))
        active_types = np.zeros(len(PokemonType))
        active_current_hp = np.zeros(1)
        active_fainted = np.zeros(1)
        active_status = np.zeros(len(Status))
        active_boosts = np.full(7, -4)
        active_effects = np.zeros(len(Effect))
        opp_active_species = np.zeros(len(GEN_DATA.pokedex))
        opp_active_types = np.zeros(len(PokemonType))
        opp_active_current_hp = np.zeros(1)
        opp_active_fainted = np.zeros(1)
        opp_active_status = np.zeros(len(Status))
        opp_active_boosts = np.full(7, -4)
        opp_active_effects = np.zeros(len(Effect))
        side_conditions = np.zeros(len(SideCondition))
        opp_side_conditions = np.zeros(len(SideCondition))
        weather = np.zeros(len(Weather))
        fields = np.zeros(len(Field))
        available_moves = np.zeros(len(GEN_DATA.moves) * 4)
        available_move_damage = np.zeros(4)
        available_switches = np.zeros(len(GEN_DATA.pokedex) * 5)
        return np.concatenate(
            [
                active_species,
                active_types,
                active_current_hp,
                active_fainted,
                active_status,
                active_boosts,
                active_effects,
                opp_active_species,
                opp_active_types,
                opp_active_current_hp,
                opp_active_fainted,
                opp_active_status,
                opp_active_boosts,
                opp_active_effects,
                side_conditions,
                opp_side_conditions,
                weather,
                fields,
                available_moves,
                available_move_damage,
                available_switches
            ],
            dtype=np.float32
        )
    
    def describe_high_embedding(self):
        active_species = np.ones(len(GEN_DATA.pokedex))
        active_types = np.ones(len(PokemonType))
        active_current_hp = np.ones(1)
        active_fainted = np.ones(1)
        active_status = np.ones(len(Status))
        active_boosts = np.full(7, 4)
        active_effects = np.ones(len(Effect))
        opp_active_species = np.ones(len(GEN_DATA.pokedex))
        opp_active_types = np.ones(len(PokemonType))
        opp_active_current_hp = np.ones(1)
        opp_active_fainted = np.ones(1)
        opp_active_status = np.ones(len(Status))
        opp_active_boosts = np.full(7, 4)
        opp_active_effects = np.ones(len(Effect))
        side_conditions = np.ones(len(SideCondition))
        opp_side_conditions = np.ones(len(SideCondition))
        weather = np.ones(len(Weather))
        fields = np.ones(len(Field))
        available_moves = np.ones(len(GEN_DATA.moves) * 4)
        available_move_damage = np.full(4, 4)
        available_switches = np.ones(len(GEN_DATA.pokedex) * 5)
        return np.concatenate(
            [
                active_species,
                active_types,
                active_current_hp,
                active_fainted,
                active_status,
                active_boosts,
                active_effects,
                opp_active_species,
                opp_active_types,
                opp_active_current_hp,
                opp_active_fainted,
                opp_active_status,
                opp_active_boosts,
                opp_active_effects,
                side_conditions,
                opp_side_conditions,
                weather,
                fields,
                available_moves,
                available_move_damage,
                available_switches
            ],
            dtype=np.float32
        )

    def describe_embedding(self) -> Space:
        low = self.describe_low_embedding()
        high = self.describe_high_embedding()
        return Box(low, high, dtype=np.float32)
    
class TrainedPlayer(Player):
    def __init__(self, model, rlplayer, *args, **kwargs):
        Player.__init__(self, *args, **kwargs)
        self.model = model
        self.rlplayer = rlplayer

    def choose_move(self, battle):
        with GRAPH.as_default():
            session = tf.compat.v1.keras.backend.get_session()
            init = tf.compat.v1.global_variables_initializer()
            session.run(init)
            state = embed_battle(battle)
            predictions = self.model.predict(np.expand_dims(np.expand_dims(state, 0), 0))
            # predictions = np.random.random((1, 1, 14834))
            action = np.argmax(predictions)
            return self.rlplayer.action_to_move(action, battle)
    
def evaluate_model(dqn: DQNAgent):
    test_opponent_random = RandomPlayer(battle_format=BATTLE_FORMAT)
    test_opponent_mbp = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT)
    test_opponent_heuristics = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT)
    test_player = RLPlayer(battle_format=BATTLE_FORMAT, opponent=test_opponent_random, start_challenging=True)
    test_player = wrap_for_old_gym_api(test_player)

    print("Results against random player:")
    dqn.test(test_player, nb_episodes=50, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {test_player.n_won_battles} victories out of {test_player.n_finished_battles} episodes"
    )
    
    test_player.reset_env(restart=True, opponent=test_opponent_mbp)
    print("Results against max base power player:")
    dqn.test(test_player, nb_episodes=50, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {test_player.n_won_battles} victories out of {test_player.n_finished_battles} episodes"
    )
    
    test_player.reset_env(restart=True, opponent=test_opponent_heuristics)
    print("Results against simple heuristics player:")
    dqn.test(test_player, nb_episodes=50, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {test_player.n_won_battles} victories out of {test_player.n_finished_battles} episodes"
    )
    test_player.close()