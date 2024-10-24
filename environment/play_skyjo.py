import skyjo_env
import numpy as np

skyjo_env_cfg = {"num_players": 3, "render_mode": "human"}
env_pettingzoo = skyjo_env.env(**skyjo_env_cfg)
env_pettingzoo.reset()

def random_admissible_policy(observation, action_mask):
    """picks randomly an admissible action from the action mask"""
    return np.random.choice(
        np.arange(len(action_mask)),
        p= action_mask/np.sum(action_mask)
    )

i_episode = 1
while i_episode <= 1:
    i_episode += 1
    env_pettingzoo.reset()
    for i, agent in enumerate(env_pettingzoo.agent_iter(max_iter=6000)):
        # get observation (state) for current agent:
        print(f"\n\n\n\n\n===================== Iteration {i} =====================")
        obs, reward, term, trunc, info = env_pettingzoo.last()

        #print("training fct:", obs, reward, term, trunc, info)
        print(f"{term = }, {trunc = }")
        # perform q-learning with update_Q_value()
        # your code here

        print(env_pettingzoo.render())

        # store current state
        # if not term and not trunc:
            # choose action using epsilon_greedy_policy()
            # your code here
        observation = obs["observations"]
        action_mask = obs["action_mask"]
        action = random_admissible_policy(observation, action_mask)

        print(f"{action_mask = }")
        print(f"sampled action {agent}: {action}")
        env_pettingzoo.step(action)
        if term:
            env_pettingzoo.step(None)
            print('done', reward)
            break
        # else:
        #     # agent is done
        #     env_pettingzoo.step(None)
        #     print('done', reward)
        #     break

#
# else:
#     print(env_pettingzoo._cumulative_rewards)