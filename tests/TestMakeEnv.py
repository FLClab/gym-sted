
import gym

env = gym.make("gym_sted:ContextualMOSTED-easy-hslb-v0")
env = gym.make("gym_sted:ContextualMOSTED-easy-hshb-v0")
env = gym.make("gym_sted:ContextualMOSTED-easy-lslb-v0")
env = gym.make("gym_sted:ContextualMOSTED-easy-lshb-v0")

env = gym.make("gym_sted:ContextualMOSTED-hard-v0")

env = gym.make("gym_sted:SequencelMOSTED-easy-hslb-v0")
env = gym.make("gym_sted:SequencelMOSTED-easy-hshb-v0")
env = gym.make("gym_sted:SequencelMOSTED-easy-lslb-v0")
env = gym.make("gym_sted:SequencelMOSTED-easy-lshb-v0")

env = gym.make("gym_sted:SequenceMOSTED-hard-v0")

print("done with success")
