DELIM = '>' # separates reversed suffixes from prefixes in GADDAG
END_WORD_DELIM = '*'

class Arc:
    ## essentially represents a path in the GADDAG

    __slots__ = "character", "destination"
    
    def __init__(self, character, destination = None):
        self.character = character
        if not destination:
            destination = State()
        self.destination = destination
    
    @property
    def letter_set(self):
        # getting the letter set from the destination set
        if self.destination:
            return self.destination.letter_set
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
        self.letter_set.add(character)
        return self.get_next(character)
    
    def add_final_arc(self, character: str, final_character: str) -> "State":
        if character not in self.arcs:
            self.arcs[character] = Arc(character)
        self.get_next(character).letter_set.add(final_character)
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

    __slots__ = "_root" # the root State of the GADDAG

    def __init__(self):
        print('\nInitializing GADDAG...\nthis may take a moment...\n')
        self._root = Gaddag.construct_root_from_txt("SOWPODS.txt")
        print('\n\nInitialized GADDAG...\n\n')

    # string input representing a filepath
    @staticmethod
    def construct_root_from_txt(filepath):
        root = State()
        with open(filepath, "r") as file:
            total_lines = sum(1 for _ in file)

        with open(filepath, "r") as file:
            for idx, line in enumerate(file):
                word = line.strip()
                word = word.upper()
                Gaddag.add_word(root, word)

                percent_complete = (idx + 1) * 100 // total_lines
                if idx and idx % (total_lines // 10) == 0:
                    print(f"{percent_complete}%")
        return root
      
    @property
    def root(self):
        return self._root
    
    
    # takes a string input of word to be added to gaddag
    @staticmethod
    def add_word(root: State, word: str):
        word = word.upper()
        # store a word path with reversed suffix > prefix
        # Example:
        # If we were to add a word "word", the arcs we want to store include:
        # >WORD, D>WOR, DR>WO, DRO>W, DROW>
        # But we also want to store arcs for all of the combinations of substrings of WORD (so repeat for WOR, ORD, OR, WO, RD)
        state = root

        for c in word[len(word):0:-1]:
            state = state.add_arc(c)
        state.add_final_arc(word[0], END_WORD_DELIM)

        for i in range(len(word)):
            suffix_reversed = word[i::-1]
            prefix = word[i + 1:]
            
            # Add reversed suffix
            state = root
            for c in suffix_reversed:
                state = state.add_arc(c)
            
            # Add delimiter and prefix
            state = state.add_arc(DELIM)
            for c in prefix:
                state = state.add_arc(c)
            state.add_arc(END_WORD_DELIM)

    # traverses the GADDAG for the given word, checking if the word is valid
    def is_word_in_gaddag(self, word: str) -> bool:
        word = word.upper()
        
        for i in range(len(word)):
            state = self._root
            
            # Check reversed prefix
            for c in word[i::-1]:
                temp_state = state.get_next(c)
                if temp_state is None:
                    break
                else:
                    state = temp_state
            else:
                # If reversed prefix succeeded, check for delimiter
                temp_state = state.get_next(DELIM)
                if temp_state is None:
                    continue # HERE
                else:
                    state = temp_state
                
                # Check for suffix
                for c in word[i + 1:]:
                    temp_state = state.get_next(c)
                    if temp_state is None:
                        break
                    else:
                        state = temp_state
                else:
                    # If suffix succeeded, check for end of word delimeter
                    if END_WORD_DELIM in state.letter_set:
                        return True
        return False
