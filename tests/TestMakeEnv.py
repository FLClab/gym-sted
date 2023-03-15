
import gym

env = gym.make("gym_sted:PreferenceMOSTED-easy-hslb-v0")
env = gym.make("gym_sted:PreferenceMOSTED-easy-hshb-v0")
env = gym.make("gym_sted:PreferenceMOSTED-easy-lslb-v0")
env = gym.make("gym_sted:PreferenceMOSTED-easy-lshb-v0")

env = gym.make("gym_sted:PreferenceMOSTED-hard-v0")

env = gym.make("gym_sted:ContextualMOSTED-easy-hslb-v0")
env = gym.make("gym_sted:ContextualMOSTED-easy-hshb-v0")
env = gym.make("gym_sted:ContextualMOSTED-easy-lslb-v0")
env = gym.make("gym_sted:ContextualMOSTED-easy-lshb-v0")

env = gym.make("gym_sted:ContextualMOSTED-hard-v0")

env = gym.make("gym_sted:SequenceMOSTED-easy-hslb-v0")
env = gym.make("gym_sted:SequenceMOSTED-easy-hshb-v0")
env = gym.make("gym_sted:SequenceMOSTED-easy-lslb-v0")
env = gym.make("gym_sted:SequenceMOSTED-easy-lshb-v0")

env = gym.make("gym_sted:SequenceMOSTED-hard-v0")

print("done with success")
