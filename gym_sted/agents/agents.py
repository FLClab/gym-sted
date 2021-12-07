
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        # Update agent from rewards

        # Samples the action
        return self.action_space.sample()
