from envs.tampnamo import *
from envs.tampbins import *
from envs.enemies import create_enemies_env, EnemiesEnvFamilyBig, EnemiesEnvFamilySmall
for diff in ["easy", "medium", "hard"]:
    for ind in range(100):
        create_enemies_env(EnemiesEnvFamilyBig, (diff, ind), globals())
        create_enemies_env(EnemiesEnvFamilySmall, (diff, ind), globals())
for ind in range(100):
    create_tampnamo_env(TAMPNAMOEnvFamily, ind, globals())
for ind in range(100):
    create_tampbins_env(TAMPBinsEnvFamily, ind, globals())
