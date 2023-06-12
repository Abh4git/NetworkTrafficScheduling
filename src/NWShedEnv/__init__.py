from gym.envs.registration import register


register(
    id="nws-v1",
    entry_point="NWShedEnv.envs:NwsEnv",
)
