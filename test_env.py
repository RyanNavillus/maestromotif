import time
import gym
import nle
print(nle.__file__, nle.__path__)
from nle import nethack
from rl_baseline.monk_tasks_nle import NetHackScoreMonk, NetHackScoreAny, NetHackScoreResetWrapper

base_actions = (
    [nethack.MiscAction.MORE] +
    list(nethack.CompassDirection) +
    list(nethack.CompassDirectionLonger) +
    list(nethack.MiscDirection) +
    [nethack.Command.KICK, nethack.Command.EAT, nethack.Command.SEARCH]
)

actions = base_actions + [
    nethack.Command.CAST,
    nethack.Command.ENHANCE,
    nethack.Command.PRAY,
    nethack.Command.QUAFF,
    nethack.Command.DROP,
]

env = NetHackScoreResetWrapper(
    options=("@.nethackrc",),
    observation_keys=[
        "tty_chars",
        "tty_colors", 
        "tty_cursor", 
        "blstats", 
        "message", 
        "glyphs", 
        "inv_strs", 
        "inv_letters", 
        "inv_oclasses"
        ],
    actions=tuple(actions),
    max_episode_steps=10000,
    penalty_mode="none",
    evaluation=False,
    eval_target="goldenexit",
)

nethack_roles = [
    "monk",
    "barbarian",
    "caveman",
    "healer",
    "knight",
    "priest",
    "ranger",
    "rogue",
    "samurai",
    "tourist",
    "valkyrie",
    "wizard",
    "archeologist",
]
create_times = []
reset_times = []
for episode in range(100):
    start = time.time()
    obs = env.reset(new_task=nethack_roles[episode % len(nethack_roles)])
    end = time.time()
    reset_times.append(end - start)
    done = False
    print(bytes(obs["message"]).decode("utf-8"))
    while not done:
        action = env.action_space.sample()  # Replace with your action selection logic
        obs, reward, done, info = env.step(action)
        # print(f"Episode: {episode}, Action: {action}, Reward: {reward}, Done: {done}")
    print("done")
print("Average reset time:", sum(reset_times) / len(reset_times))
# env = NetHackScoreAny()

# for episode in range(10):
#     obs = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()  # Replace with your action selection logic
#         obs, reward, done, info = env.step(action)
#         # print(f"Episode: {episode}, Action: {action}, Reward: {reward}, Done: {done}")
