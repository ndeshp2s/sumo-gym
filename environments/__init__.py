from gym.envs.registration import register

register(
    id='Highway-v0',
    entry_point='environments.highway.highway_env:HighwayEnv',
)
