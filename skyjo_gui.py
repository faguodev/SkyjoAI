import tkinter as tk
from tkinter import messagebox
from functools import partial
import random
from environment.skyjo_game import SkyjoGame

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from environment.skyjo_env import env as skyjo_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import logging

from models.action_mask_model import TorchActionMaskModel

def random_admissible_policy(obs):
    observation = obs["observations"]
    action_mask = obs["action_mask"]
    admissible_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
    return random.choice(admissible_actions)

def pre_programmed_smart_policy(obs):
    observation = obs["observations"]
    action_mask = obs["action_mask"]
    admissible_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
    #check wether card still has to be taken from discard pile or draw pile
    if 24 in admissible_actions:
        #choose wether to take the upper most card from the discard pile or choose a random from the draw pile

        #if card on discard pile has value smaller equal 3 take it
        if 1 in observation[17:23]:
            action = 25
        #else draw random card from draw pile
        else:
            action = 24
    #if card was already taken from deck/discard pile continue with placing/throwing away
    else:
        #go through one-hot-encoded hand cards to find value of hand card
        for i, hand in enumerate(observation[34:51]):
            if hand == 1:
                hand_card_value = i - 2
        #find position and highest value of players cards (here unknown cards are valued as 5)
        max_card_value = -2
        masked_cards = []
        for i in range(12):
            idx_start = i*17+51
            #find value of current card (17th one-hot-encoded field is for refunded cards and therefore ignored)
            for j, val in enumerate(observation[idx_start:idx_start+16]):
                if val == 1:
                    if j == 15:
                        masked_cards.append(i)
                        if max_card_value < 5:
                            max_card_value = 5
                            imax = i
                    elif max_card_value < j - 2:
                        max_card_value = j-2
                        imax = i
        #1st case hand card value is lower equal than 3 (if card was taken from discard this branch will be taken for 100%)
        #place card on position with max_card_value
        if hand_card_value <= 3:
            action = imax
        #else if hand is smaller than max_card_value replace card with higest value with handcard
        elif hand_card_value < max_card_value:
            action = imax
        #else throw hand card away and reveal masked card
        else:
            action = 12 + masked_cards[0]
    return action

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RewardDecayCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Decay the reward scaling factor over training iterations
        action_reward_decay = max(0.05, 1.0 - result["training_iteration"] * 0.005)
        # env = algorithm.workers.local_worker().env
        # env = algorithm.workers.local_env_runner.env
        algorithm.config.env_config["action_reward_decay"] = action_reward_decay
        logger.info(action_reward_decay)

skyjo_config_old = {
    "num_players": 3,
    "score_penalty": 2.0,
    "observe_other_player_indirect": False,
    "mean_reward": 1.0,
    "reward_refunded": 10,
    "final_reward": 100,
    "score_per_unknown": 5.0,
    "action_reward_decay": 1.0,
    "old_reward": True,
    "render_mode": "human",
}

model_config = {
    "custom_model": TorchActionMaskModel,
    # Add the following keys:
    "fcnet_hiddens": [1024, 1024, 1024, 512, 512],
    "fcnet_activation": "relu",
}

def env_creator(config):
    return PettingZooEnv(skyjo_env(**config))

register_env("skyjo", env_creator)

test_env = env_creator(skyjo_config_old)
obs_space = test_env.observation_space
act_space = test_env.action_space

def policy_mapping_fn(agent_id, _, **kwargs):
    return "policy_" + str(agent_id) #int(agent_id.split("_")[-1])

config_old = (
    PPOConfig()
    .training(model=model_config, )
    .environment("skyjo", env_config=skyjo_config_old)
    .framework('torch')
    .callbacks(RewardDecayCallback)
    .env_runners(num_env_runners=1)
    # .rollouts(num_rollout_workers=6)
    .resources(num_gpus=0)
    .multi_agent(
        policies={
            "policy_0": (None, obs_space[0], act_space[0], {"entropy_coeff":0}),
            "policy_1": (None, obs_space[1], act_space[1], {"entropy_coeff":0}),
            "policy_2": (None, obs_space[2], act_space[2], {"entropy_coeff":0})
        },
        policy_mapping_fn=policy_mapping_fn,#(lambda agent_id, *args, **kwargs: agent_id),
    )
    .evaluation(evaluation_num_env_runners=0)
    .debugging(log_level="INFO")
)

algo_old = config_old.build()
#model_save_dir_old = "v3_trained_models_old_rewards_0.03_0.03_0.03_false"
#final_dir_old = model_save_dir_old + f"/checkpoint_100"
#algo_old.restore(final_dir_old)

def policy_two(obs):
    policy = algo_old.get_policy(policy_id=policy_mapping_fn(0, None))
    action_exploration_policy, _, action_info = policy.compute_single_action(obs)
    # 
    action = action_exploration_policy
    return action


def policy_one(obs):
    policy = algo_old.get_policy(policy_id=policy_mapping_fn(1, None))
    action_exploration_policy, _, action_info = policy.compute_single_action(obs)
    # 
    action = action_exploration_policy
    return action







class SkyjoGUI:
    def __init__(self, num_players=4, player_types=None, observe_other_players_indirect=False):
        """
        Initialize the Skyjo game GUI.
        :param num_players: Number of players (max 4)
        :param player_types: List indicating if a player is 'human' or an AI function
        """
        if player_types is None:
            player_types = ['human'] * num_players
        assert len(player_types) == num_players, "Player types must match the number of players."
        self.num_players = num_players
        self.player_types = player_types
        self.game = SkyjoGame(num_players=num_players, observe_other_player_indirect=observe_other_players_indirect)
        self.root = tk.Tk()
        self.root.title("Skyjo Game")
        self.root.configure(bg='white')  # Set background to white
        self.card_width = 6  # Adjusted for larger text
        self.card_height = 3
        self.create_widgets()
        self.current_player = None
        self.selection = None  # 'discard_pile', 'hand_card', or None
        self.hand_card_button.config(state='disabled')
        self.update_ui()
        self.root.mainloop()

    def create_widgets(self):
        # Create frame for piles and hand card
        self.piles_frame = tk.Frame(self.root, bg='white')
        self.piles_frame.grid(row=0, column=0, columnspan=self.num_players, padx=5, pady=5)

        # Discard pile button
        self.discard_pile_button = tk.Button(self.piles_frame, text="", command=self.take_discard_pile,
                                             width=self.card_width, height=self.card_height, font=('Arial', 12))
        self.discard_pile_button.grid(row=0, column=0, padx=5)

        # Draw pile button
        self.draw_pile_button = tk.Button(self.piles_frame, text="", command=self.draw_new_card,
                                          width=self.card_width, height=self.card_height, font=('Arial', 12))
        self.draw_pile_button.grid(row=0, column=1, padx=5)

        # Hand card button
        self.hand_card_button = tk.Button(self.piles_frame, text="", command=self.toggle_hand_card_selection,
                                          width=self.card_width, height=self.card_height, font=('Arial', 12), state='disabled')
        self.hand_card_button.grid(row=0, column=2, padx=5)

        # Create frames for each player's cards
        self.player_frames = []
        for i in range(self.num_players):
            frame = tk.LabelFrame(self.root, text=f"Player {i+1}", bg='white', font=('Arial', 12))
            frame.grid(row=2, column=i, padx=5, pady=5, sticky='n')
            self.player_frames.append(frame)

        # Action explanation
        self.action_label = tk.Label(self.root, text="Game Started", bg='white', font=('Arial', 12))
        self.action_label.grid(row=3, column=0, columnspan=self.num_players)

    def update_player_grid(self, player_idx):

        frame = self.player_frames[player_idx]
        # Clear the frame
        for widget in frame.winfo_children():
            widget.destroy()
        # Display the player's cards
        for idx, (card_value, mask_value) in enumerate(zip(self.game.players_cards[player_idx], self.game.players_masked[player_idx])):
            card_text = ""
            if mask_value == 2:
                card_text = "?"
                bg_color, fg_color = "gray", "black"
            elif mask_value == 1:
                card_text = str(card_value)
                bg_color, fg_color = self.get_card_color(card_value, mask_value)
            elif mask_value == 0:
                card_text = "X"
                bg_color, fg_color = "white", "black"
            btn = tk.Button(frame, text=card_text, bg=bg_color, fg=fg_color,
                            width=self.card_width, height=self.card_height, font=('Arial', 12))
            if self.game.expected_action[0] == player_idx:
                # Enable buttons based on selection
                if self.current_player == player_idx and self.player_grid_enabled and mask_value != 0:
                    btn.config(state='normal')
                else:
                    btn.config(state='disabled')
                btn.config(command=partial(self.place_card, idx))
            else:
                btn.config(state='disabled')
            btn.grid(row=idx % 3, column=idx // 3, padx=2, pady=2)



    def update_ui(self):
        # Update the UI elements based on the game state

        for i in range(self.num_players):
            self.update_player_grid(i)

        # Update discard pile button
        if self.game.discard_pile:
            top_discard = self.game.discard_pile[-1]
            bg_color, fg_color = self.get_card_color(top_discard, mask_value=1)
            self.discard_pile_button.config(text=str(top_discard), bg=bg_color, fg=fg_color)
        else:
            # Empty discard pile
            self.discard_pile_button.config(text="Empty", bg='white', fg='black')

        # Update draw pile button
        # Since we don't know the top card, we can use a placeholder
        self.draw_pile_button.config(text="Draw Pile", bg='white', fg='black')

        # Update hand card button
        if self.game.hand_card != self.game.fill_masked_unknown_value:
            hand_card_value = self.game.hand_card
            bg_color, fg_color = self.get_card_color(hand_card_value, mask_value=1)
            self.hand_card_button.config(text=str(hand_card_value), bg=bg_color, fg=fg_color, state='normal')
            if self.selection == 'hand_card':
                self.hand_card_button.config(relief='solid', bd=3, highlightbackground='purple', highlightcolor='purple')
            else:
                self.hand_card_button.config(relief='raised', bd=2)
        else:
            # Empty hand card
            self.hand_card_button.config(text="Empty", bg='white', fg='black', state='disabled')
            self.hand_card_button.config(relief='raised', bd=2)

        # Highlight selection
        if self.selection == 'discard_pile':
            self.discard_pile_button.config(relief='solid', bd=3, highlightbackground='purple', highlightcolor='purple')
        else:
            self.discard_pile_button.config(relief='raised', bd=2)

        # Update action label
        next_player = self.game.expected_action[0]
        next_action = self.game.expected_action[1]
        self.action_label.config(text=f"Player {next_player+1}'s turn to {next_action}")

        # Check if the game is over
        if self.game.has_terminated:
            scores = self.game.game_metrics["final_score"]
            winner = scores.index(min(scores)) + 1
            messagebox.showinfo("Game Over", f"Game Over! Player {winner} wins!\nScores: {scores}")
            self.root.quit()
            return

        # Proceed to the next action
        self.current_player = next_player

        if self.player_types[next_player] != 'human':
            # AI player's turn
            self.root.after(1000, self.ai_turn)
        else:
            # Enable/disable buttons based on expected action
            if next_action == 'draw':
                self.enable_piles()
                if self.selection != 'discard_pile':
                    self.player_grid_enabled = False
                    self.update_player_grid(next_player)
            elif next_action == 'place':
                self.disable_piles()
                self.player_grid_enabled = True
                self.update_player_grid(next_player)
    def get_card_color(self, value, mask_value):
        # Determine the background and foreground color based on the card value
        if mask_value == 2:
            return "gray", "black"
        if value in [-2, -1]:
            return "darkblue", "white"
        elif value == 0:
            return "lightblue", "black"
        elif 1 <= value <= 4:
            return "#90EE90", "black"  # Lighter green
        elif 5 <= value <= 8:
            return "yellow", "black"
        elif 9 <= value <= 12:
            return "#FF7F7F", "black"  # Lighter red
        else:
            return "white", "black"

    def enable_piles(self):
        self.discard_pile_button.config(state='normal')
        self.draw_pile_button.config(state='normal')

    def disable_piles(self):
        self.discard_pile_button.config(state='disabled')
        self.draw_pile_button.config(state='disabled')

    def disable_all_piles(self):
        self.disable_piles()
        self.hand_card_button.config(state='disabled')

    def enable_player_grid(self):
        self.player_grid_enabled = True
    def disable_player_grid(self):
        self.player_grid_enabled = False

    def take_discard_pile(self):
        if self.game.expected_action[1] == 'draw' and self.current_player is not None:
            if self.selection == 'discard_pile':
                # Deselect
                self.selection = None
                self.discard_pile_button.config(relief='raised', bd=2)
                self.disable_player_grid()
                self.update_ui()
            else:
                # Select discard pile
                self.selection = 'discard_pile'
                self.discard_pile_button.config(relief='solid', bd=3, highlightbackground='purple', highlightcolor='purple')
                self.enable_player_grid()
                self.update_ui()
        else:
            messagebox.showwarning("Invalid Action", "It's not your turn to draw.")

    def draw_new_card(self):
        if self.game.expected_action[1] == 'draw' and self.current_player is not None:
            self.selection = "hand_card"
            self.perform_action(24)  # Draw from draw pile
            self.hand_card_button.config(relief='solid', bd=3, highlightbackground='purple', highlightcolor='purple')
            self.enable_player_grid()
            self.discard_pile_button.config(state='disabled')
            self.update_ui()
        else:
            messagebox.showwarning("Invalid Action", "It's not your turn to draw.")

    def toggle_hand_card_selection(self):
        if self.selection == 'hand_card':
            # Deselect hand card
            self.selection = None
            self.hand_card_button.config(relief='raised', bd=2)
        else:
            # Select hand card
            self.selection = 'hand_card'
            self.hand_card_button.config(relief='solid', bd=3, highlightbackground='purple', highlightcolor='purple')

    def place_card(self, idx):
        if self.game.expected_action[1] == 'draw' and self.current_player is not None and self.selection != "discard_pile":
            messagebox.showwarning("Invalid Action", "You need to select a pile first.")
            return
        
        if self.game.expected_action[1] == 'draw' and self.current_player is not None and self.selection == "discard_pile":
            # Take discard pile card and place it
            try:
                self.perform_action(25)  # Draw from discard pile
                action_int = idx  # Place the hand card on this position
                self.perform_action(action_int)
                self.selection = None
                self.discard_pile_button.config(relief='raised', bd=2)
                self.disable_player_grid()
                self.discard_pile_button.config(state='disabled')
                self.draw_pile_button.config(state='disabled')
                self.hand_card_button.config(state='disabled')
                self.update_ui()
            except Exception as e:
                messagebox.showerror("Error", str(e))
            return

        if self.game.expected_action[1] == 'place' and self.current_player is not None:
            if self.selection == 'hand_card':
                # Take new card from draw pile
                try:
                    self.perform_action(idx)
                    self.selection = None
                    self.hand_card_button.config(relief='raised', bd=2)
                    self.disable_player_grid()
                    self.discard_pile_button.config(state='disabled')
                    self.draw_pile_button.config(state='disabled')
                    self.hand_card_button.config(state='disabled')
                    self.update_ui()
                except Exception as e:
                    messagebox.showerror("Error", str(e))
            else:
                # Reveal the card and discard hand card
                action_int = idx + 12  # Discard hand card and reveal this card
                self.perform_action(action_int)
                self.disable_player_grid()
                self.discard_pile_button.config(state='disabled')
                self.draw_pile_button.config(state='disabled')
                self.hand_card_button.config(state='disabled')
                self.update_ui()
        else:
            messagebox.showwarning("Invalid Action", "It's not your turn to place.")

    def perform_action(self, action_int):
        try:
            game_over, last_action = self.game.act(self.current_player, action_int)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def ai_turn(self):
        obser, mask = self.game.collect_observation(self.current_player)
        obs = {
            "observations": obser,
            "action_mask": mask
        }
        ai_policy = self.player_types[self.current_player]
        action_int = ai_policy(obs)
        self.perform_action(action_int)
        self.update_ui()

    # Additional methods to manage UI state
    @property
    def player_grid_enabled(self):
        return getattr(self, '_player_grid_enabled', False)

    @player_grid_enabled.setter
    def player_grid_enabled(self, value):
        self._player_grid_enabled = value

# Example usage:
if __name__ == "__main__":
    # Define player types: 'human' or an AI function
    player_types = [
        pre_programmed_smart_policy,
        random_admissible_policy,
        #policy_one,
        'human',
    ]
    # Replace 'human' with random_admissible_policy to make all AI players
    gui = SkyjoGUI(num_players=3, player_types=player_types, observe_other_players_indirect=True)
