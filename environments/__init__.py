from gym.envs.registration import register

register(
    id='Highway-v0',
    entry_point='environments.highway.highway_env:HighwayEnv',
)

register(
    id='Urban-v0',
    entry_point='environments.urban.urban_env:UrbanEnv',
)