import matplotlib
import matplotlib.pyplot as plt
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

log_file = 'result/test_log.txt'

class Atari(object):
    def __init__(self, env_name="FreewayDeterministic-v4", agent_history_length=4):
        self.env = gym.make(env_name)
        self.state = None
        self.agent_history_length = agent_history_length

    def reset(self):
        observation = self.env.reset()
        frame = self.image_proc(observation).to(device)
        self.state = frame.repeat(1,4,1,1)
        return self.state

    def image_proc(self, image):
        return frame_proc(image)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        frame = self.image_proc(observation).to(device)
        next_state = torch.cat( (self.state[:, 1:, :, :], frame.unsqueeze(0)), axis=1 )
        self.state = next_state
        return next_state, reward, done, info

    def render(self):
        observation = self.env.render(mode='rgb_array')
        return observation

class LearntAgent(object):
    def __init__(self):
        self.action_dim = 3
        self.net = CNN(84, 84, self.action_dim).to(device)
        self.interaction_steps = 0

    def evaluate_action(self, state, rand=0.1):
        if random.random() < rand:
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long), np.array([[0., 0., 0.,]])
        with torch.no_grad():
            q_value = self.net(state)
            return q_value.max(1)[1].view(1, 1), q_value.cpu().numpy()

    def restore_model(self, path):
        self.net.load_state_dict(torch.load(path, map_location=device))
        self.net.eval()
        print("[Info] Restore model from '%s' !"%path)


def play(path):
    agent = LearntAgent()
    env = Atari()

    agent.restore_model(path)
    log_fd = open(log_file, 'w')
    for i_episode in range(10):
        episode_reward = 0
        state = env.reset()

        fig = plt.figure()
        fig.set_facecolor('w')
        img0 = plt.imshow(env.render())

        for s in range(10000):
            #env.render()
            action, q_value = agent.evaluate_action(state)

            plt.title( "Step: %04d\nNOPE: %f, UP: %f, DOWN: %f\nAverage Q: %f, Action: %d\n"\
                        %(s+1, q_value[0,0], q_value[0,1], q_value[0,2], np.average(q_value), action.item()) )
            img0.set_data(env.render())
            display.display(plt.gcf())
            display.clear_output(wait=True)

            next_state, reward, done, _ = env.step(action.item())
            state = next_state
            episode_reward += reward

            if done:
                print("Episode: %6d, interaction_steps: %6d, reward: %2d, epsilon: %f"%(i_episode, agent.interaction_steps, episode_reward, 0.1))
                log_fd.write("Episode: %6d, interaction_steps: %6d, reward: %2d, epsilon: %f"%(i_episode, agent.interaction_steps, episode_reward, 0.1))
                break
    log_fd.close()
model_path = "./HW3/model/q_policy_checkpoint_1538048.pth"
play(model_path)
