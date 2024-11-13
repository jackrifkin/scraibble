import scrabble_gym as sg

env = sg.ScrabbleEnv()

def display_help():
    print("""
            Commands:
            - show_board: Display the current state of the board.
            - show_rack: Display your current letter rack.
            - place [row] [col] [direction] [word] [rack_letter_indices]: Place a whole word on the board. 
                  Example: place 7 7 right HELLO 0 1 2 3 4
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
            env.show_rack()

        elif command.startswith("place"):
            try:
                # Parsing the place command
                _, row, col, direction, word, *rack_letter_indices = command.split()
                row, col = int(row), int(col)
                rack_letter_indices = list(map(int, rack_letter_indices))
                down = direction == "down"  # Adjust to check for 'down' as direction

                # Place the word on the board
                success = env.place_word(row, col, down, word, rack_letter_indices)

                if success:
                    print(f"Placed word '{word}'.")
                else:
                    print("Failed to place the word. Check the board and try again.")

                env.render()

            except ValueError:
                print("Invalid place command format. Use: place [row] [col] [direction] [word] [rack_letter_indices]")

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
