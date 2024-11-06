import tkinter as tk
from tkinter import messagebox
from functools import partial
import numpy as np
from typing import Callable, Optional
import random
from skyjo_game import SkyjoGame

# Ensure the SkyjoGame class is in the same directory or adjust the import accordingly
# from skyjo_game_logic import SkyjoGame

# Placeholder AI policy for AI players
def random_admissible_policy(observation, action_mask):
    admissible_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
    return random.choice(admissible_actions)

class SkyjoGUI:
    def __init__(self, num_players=4, player_types=None):
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
        self.game = SkyjoGame(num_players=num_players)
        self.root = tk.Tk()
        self.root.title("Skyjo Game")
        self.root.configure(bg='white')  # Set background to white
        self.card_width = 6  # Adjusted for larger text
        self.card_height = 3
        self.create_widgets()
        self.current_player = None
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
        self.hand_card_button = tk.Button(self.piles_frame, text="", state='disabled',
                                          width=self.card_width, height=self.card_height, font=('Arial', 12))
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

    def update_ui(self):
        # Update the UI elements based on the game state
        for i in range(self.num_players):
            frame = self.player_frames[i]
            # Clear the frame
            for widget in frame.winfo_children():
                widget.destroy()
            # Display the player's cards
            for idx, (card_value, mask_value) in enumerate(zip(self.game.players_cards[i], self.game.players_masked[i])):
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
                if self.game.expected_action[0] == i and self.game.expected_action[1] == 'place':
                    # Only bind if it's the current player's turn and action is 'place'
                    btn.config(command=partial(self.place_card, idx))
                btn.grid(row=idx // 4, column=idx % 4, padx=2, pady=2)  # Adjusted for 3 rows and 4 columns

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
            self.hand_card_button.config(text=str(hand_card_value), bg=bg_color, fg=fg_color)
        else:
            # Empty hand card
            self.hand_card_button.config(text="Empty", bg='white', fg='black')

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
                self.draw_pile_button.config(state='normal')
                self.discard_pile_button.config(state='normal')
            else:
                self.draw_pile_button.config(state='disabled')
                self.discard_pile_button.config(state='disabled')

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

    def take_discard_pile(self):
        if self.game.expected_action[1] == 'draw' and self.current_player is not None:
            action_int = 25  # Action code for taking from discard pile
            self.perform_action(action_int)
        else:
            messagebox.showwarning("Invalid Action", "It's not your turn to draw.")

    def draw_new_card(self):
        if self.game.expected_action[1] == 'draw' and self.current_player is not None:
            action_int = 24  # Action code for drawing from draw pile
            self.perform_action(action_int)
        else:
            messagebox.showwarning("Invalid Action", "It's not your turn to draw.")

    def place_card(self, idx):
        if self.game.expected_action[1] == 'place' and self.current_player is not None:
            if self.game.hand_card != self.game.fill_masked_unknown_value:
                action_int = idx  # Place the hand card on this position
            else:
                action_int = idx + 12  # Discard hand card and reveal a new card
            self.perform_action(action_int)
        else:
            messagebox.showwarning("Invalid Action", "It's not your turn to place.")

    def perform_action(self, action_int):
        try:
            game_over, last_action = self.game.act(self.current_player, action_int)
            self.update_ui()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def ai_turn(self):
        obs, action_mask = self.game.collect_observation(self.current_player)
        ai_policy = self.player_types[self.current_player]
        action_int = ai_policy(obs, action_mask)
        self.perform_action(action_int)

# Example usage:
if __name__ == "__main__":
    # Define player types: 'human' or an AI function
    player_types = [
        'human',
        random_admissible_policy,
        random_admissible_policy,
        random_admissible_policy
    ]
    # Replace 'human' with random_admissible_policy to make all AI players
    gui = SkyjoGUI(num_players=4, player_types=player_types)
