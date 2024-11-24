DELIM = '>' # separates reversed suffixes from prefixes in GADDAG

class Arc:
    ## essentially represents a path in the GADDAG

    __slots__ = "character", "destination"
    
    def __init__(self, character, destination):
        self.character = character
        if not destination:
            destination = State()
        self.destination = destination
    
    @property
    def letter_set(self):
        # getting the letter set from the destination set
        if self.destination:
            return self.destinaton.letter_set
        else:
            return None
    
    def get_next(self, character):
        if character in self.destination.arcs:
            return self.destination.arcs[character]
        else:
            return None


class State:
    ## represents the different paths that can be taken from this specific state  

    __slots__ = "arcs", "letter_set"

    def __init__(self):
        self.arcs = dict()
        self.letter_set = set()

    def __iter__(self):
        for char in self.arcs:
            yield self.arcs[char]
    
    def __contains__(self, char):
        return char in self.arcs

    # destination is a State
    # if no destination, the default destination is empty new State
    def add_arc(self, character: str, destination: "State" = None) -> "State":
        if character not in self.arcs:
            self.arcs[character] = Arc(character, destination)
        return self.get_next(character)
    
    def add_final_arc(self, character: str, final_character: str) -> "State":
        if character not in self.arcs:
            self.arcs[character] = Arc(character, State())
        self.get_next(character).add_to_letter_set(final_character)
        return self.get_next(character)

    # gets the next node that this character leads to, which is a State
    def get_next(self, character):
        if character in self.arcs:
            return self.arcs[character].destination 
        else:
            return None
    
    # gets the corresponding arc from this state, if it exists
    def get_arc(self, character):
        if character in self.arcs:
            return self.arcs[character]
        else:
            return None

    # adds a letter to the letter set
    def add_to_letter_set(self, character):
        self.letter_set.add(character)


class Gaddag:
    ## represents a bidirectional acyclic word graph to traverse in order to find playable words

    __slots__ = "root" # the root State of the GADDAG

    def __init__(self):
        self.root = State()
        self.construct_from_txt("SOWPODS.txt")
    
    # string input representing a filepath
    @classmethod
    def construct_from_txt(cls, filepath):
        root = State()
        with open(filepath, "r") as file:
            for line in file:
                word = line.strip()
                word = word.upper()
                Gaddag.add_word(root, word)
        return cls(root)
    
    # takes a string input of word to be added to gaddag
    @staticmethod
    def add_word(root, word):
        # store a word path with reversed suffix > prefix
        # Example:
        # If we were to add a word "word", the arcs we want to store include:
        # >WORD, D>WOR, DR>WO, DRO>W, DROW>
        # But we also want to store arcs for all of the combinations of substrings of WORD (so repeat for WOR, ORD, OR, WO, RD)
        state = root
        # gets from the last character to the third character in the word (inclusive)
        for c in reversed(word[2:]):
            state = state.add_arc(c) ## TODO - double check if passing in no destination is okay
        state.add_final_arc(word[1], word[0])

        # gets from the second last character to the first character in the word (ex: ROW for WORD)
        state = root
        for c in reversed(word[:len(word) - 1]):
            state = state.add_arc(c)
        state.add_final_arc(DELIM, word[-1])

        # for each substring from second to last to first character (reversed)
        # essentially building in reverse order
        for i in range(len(word) - 2, 0, -1):
            destination = state
            state = root
            for c in reversed(word[i - 1]):
                state = state.add_arc(c)
            state = state.add_arc(DELIM)
            # make state for next iteration at the second to last node, and the destination is this state
            state.add_arc(word[i], destination) 