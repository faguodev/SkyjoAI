import tkinter as tk
from tkinter import messagebox
from functools import partial
import random
from environment.skyjo_game import SkyjoGame

from environment.skyjo_env import env as skyjo_env
import logging

import numpy as np
from stable_baselines3 import PPO

# This file is used to allow playing against the model trained in the
# github repository https://github.com/Guillaume-Barthe/Skyjo. They used
# only single agent training in a simplified environment.




##############################
# GLOBAL STATE: TURN COUNTER
##############################
global_turn_count = 0

##############################
# LOAD YOUR SB3 MODEL
##############################
# Suppose you have a model saved at "PPO_1M_multi.zip"
sb3_model_path = "PPO_1M_multi.zip"
model_env2 = PPO.load(sb3_model_path)


##############################
# OBSERVATION MAPPING
##############################
def obs_one_to_obs_two(obs_one: dict, turn_counter: int) -> np.ndarray:
    """
    Convert Environment One observation dict into an Environment Two–style observation.

    For a 2-player game in Environment One (PettingZoo):
      obs_one["observations"] might have length 43:
        - Indices:
          3 -> discard_top
          4 -> deck_card
          19..31 -> 12 cards for player0
          31..43 -> 12 cards for player1

    We’ll build a single array of length 27:
      [discard_top, turn_counter, deck_card, 12 board vals (player0), 12 board vals (player1)]

    Hidden/refunded cards = 15 or -14 => we turn them into 5.0 for environment Two.
    """
    obs_vec = obs_one["observations"]
    print("obs_vec")
    print(obs_vec)
    discard_top = obs_vec[17]
    deck_card   = obs_vec[18]

    # Grab the 12 cards for player0
    raw_board_p0 = obs_vec[19:31]
    board_int_p0 = []
    for val in raw_board_p0:
        if val == 15:
            board_int_p0.append(5.0)
        
        elif val == -14:  # Hidden/refunded
            board_int_p0.append(0.0)
        else:
            board_int_p0.append(float(val))

    # Grab the 12 cards for player1
    raw_board_p1 = obs_vec[31:43]
    board_int_p1 = []
    for val in raw_board_p1:
        if val == 15 or val == -14:
            board_int_p1.append(5.0)
        else:
            board_int_p1.append(float(val))

    state = float(turn_counter)

    obs_env2 = np.concatenate((
        np.array([discard_top, state, deck_card], dtype=np.float32),
        np.array(board_int_p0, dtype=np.float32),
        np.array(board_int_p1, dtype=np.float32),
    ))

    print("obs_env")
    print(obs_env2)

    return obs_env2


##############################
# ACTION MAPPING
##############################
def act_two_to_act_one(action_two: np.ndarray, expected_action: str, obs_one: dict) -> int:
    """
    Convert the (2,) action from environment Two to a single integer action for environment One.

    action_two[0]: 0=draw from deck, 1=take from discard
    action_two[1]: 0..11=place at position, 12=discard & reveal

    Environment One action mapping:
    - expected_action="draw": 
        if draw_or_take=0 -> 24 (draw from deck)
        if draw_or_take=1 -> 25 (take from discard)
    - expected_action="place":
        if place_or_discover<12 -> place that position (0..11)
        if place_or_discover=12 -> reveal a random masked card:
            valid reveal actions are 12..23, filtered by action_mask.
    """
    draw_or_take = action_two[0]
    place_or_discover = action_two[1]

    if expected_action == "draw":
        global global_turn_count
        global_turn_count += 1
        # Draw step
        if draw_or_take == 0:
            return 24  # draw from draw pile
        else:
            return 25  # take from discard pile

    elif expected_action == "place":

        # Place step
        if place_or_discover < 12:
            # place hand card onto that position
            return place_or_discover
        else:
            # Need to reveal a card: pick a random masked card action from 12..23
            action_mask = obs_one["action_mask"]
            reveal_actions = [a for a in range(12, 24) if action_mask[a] == 1]
            if len(reveal_actions) == 0:
                # fallback if none available (should not happen)
                return 12
            else:
                return random.choice(reveal_actions)

    # fallback
    return 24


##############################
# POLICY FUNCTION
##############################
def sb3_policy_env2(obs_one: dict, game) -> int:
    """
    Policy function integrating the SB3 model from environment Two into environment One.

    Steps:
       1) Convert PettingZoo obs -> env2 obs (with turn count)
       2) model.predict(obs_env2)
       3) convert env2 action -> env1 action
    """
    global global_turn_count

    obs_env2 = obs_one_to_obs_two(obs_one, global_turn_count)
    action_two, _ = model_env2.predict(obs_env2, deterministic=True)

    next_action = game.get_expected_action()[1]  # "draw" or "place"
    action_one = act_two_to_act_one(action_two, next_action, obs_one)
    return action_one


###################################################################
# 5) USE THIS POLICY IN YOUR SKYjoGUI CODE (Environment One)
###################################################################
#
# Inside your `if __name__ == "__main__":` block (or wherever you configure GUI):
#
# from environment.skyjo_game import SkyjoGame
# from your_module_above import sb3_policy_env2
#
# player_types = [
#     sb3_policy_env2,  # AI using stable-baselines model from env Two
#     'human',          # or another AI, or 'human'
# ]
#
# gui = SkyjoGUI(num_players=2, player_types=player_types, observe_other_players_indirect=True)
#
###################################################################
# 6) HOW THE GUI CALLS IT
###################################################################
#
# In SkyjoGUI.ai_turn(), you do something like:
#    obser, mask = self.game.collect_observation(self.current_player)
#    obs_dict = {"observations": obser, "action_mask": mask}
#    action_int = self.player_types[self.current_player](obs_dict, self.game)
#    self.perform_action(action_int)
#
# That’s all. Now your SB3 agent from environment Two is playing inside environment One!



























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
            scores = [int(x) for x in self.game.game_metrics["final_score"]]
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
        action_int = ai_policy(obs, self.game)
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
        sb3_policy_env2,
        #policy_one,
        'human',
    ]
    # Replace 'human' with random_admissible_policy to make all AI players
    gui = SkyjoGUI(num_players=2, player_types=player_types, observe_other_players_indirect=False)
