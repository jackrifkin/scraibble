import scrabble_gym as sg

env = sg.ScrabbleEnv()

def display_help():
    print("""
            Commands:
            - show_board: Display the current state of the board.
            - show_rack: Display your current letter rack.
            - place [row] [col] [letter]: Place a letter on the board. 
                  Example: place 7 7 0   (places the letter corresponding to 0 on your rack at row 7, col 7)
            - reset: Reset the game.
            - exit: Exit the game.
        """)


def main():
    print("Welcome to the Scrabble CLI!")
    env.reset()

    while True:
        display_help()
        command = input("Enter command: ").strip().lower()
        if command == "show_board":
            env.render()
        elif command == "show_rack":
            print("Letter Rack:", env.letter_rack)
        elif command.startswith("place"):
            _, row, col, letter = command.split()
            row, col, letter = int(row), int(col), int(letter)

            # Perform the step in the environment
            obs, reward, done, info = env.step((row, col, letter))

            # Check for invalid move
            if reward == -1:
                print("Invalid move. Try again.")
            else:
                print(f"Placed letter. Reward: {reward}")
                env.render()
        elif command == "reset":
            env.reset()
            print("Game reset.")
            env.render()
        elif command == "exit":
            print("Exiting game.")
            break
        else:
            print("Unknown command.")
            display_help()


if __name__ == "__main__":
    main()
