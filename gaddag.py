import json

class Gaddag:
    class Node:
        BREAK = 62  # '>' in ASCII
        EOW = 36    # '$' in ASCII
        ROOT = 32   # ' ' in ASCII
        
        def __init__(self, value):
            self.value = value
            self.children = {}

        def contains_key(self, value):
            return value in self.children

        def add_child(self, value, node=None):
            if value not in self.children:
                self.children[value] = node or Gaddag.Node(value)
            return self.children[value]

    def __init__(self):
        self.root_node = Gaddag.Node(Gaddag.Node.ROOT)

    def add(self, word):
        word = word.lower()
        prev_node = []
        chars = list(word.encode('utf-8'))
        
        for i in range(len(chars)):
            word_chars = chars[:i+1][::-1] + [Gaddag.Node.BREAK] + chars[i+1:] + [Gaddag.Node.EOW]
            current_node = self.root_node
            break_found = False
            j = 0

            for c in word_chars:
                if break_found and j < len(prev_node):
                    current_node.add_child(c, prev_node[j])
                    break
                current_node = current_node.add_child(c)
                if j == len(prev_node):
                    prev_node.append(current_node)
                if c == Gaddag.Node.BREAK:
                    break_found = True
                j += 1

    @staticmethod
    def get_word(char_array):
        try:
            node_break_index = char_array.index(Gaddag.Node.BREAK)
            new_bytes = char_array[:node_break_index][::-1] + char_array[node_break_index + 1:]
            return bytes(new_bytes).decode('utf-8').lower()
        except ValueError:
            return ""

    def word_defined(self, word):
        return word.lower() in self.find_words("", word)

    def find_words(self, hook, rack):
        hook = list(hook.lower().encode('utf-8'))[::-1]
        rack = list(rack.lower().encode('utf-8'))
        letters = []
        found_words = set()
        Gaddag.find_words_recursive(self.root_node, found_words, letters, rack, hook)
        return list(found_words)

    ## rack is the letters you have in your hand
    ## hook is a letter on the board that you can hook onto
    ## do not need to call on this method, just call find words
    @staticmethod
    def find_words_recursive(node, rtn, letters, rack, hook):
        if node.value == Gaddag.Node.EOW:
            word = Gaddag.get_word(letters)
            if word not in rtn:
                rtn.add(word)
            return
        
        new_letters = letters + ([node.value] if node.value != Gaddag.Node.ROOT else [])

        if hook:
            if node.contains_key(hook[0]):
                next_hook = hook[1:]
                next_node = node.children[hook[0]]
                Gaddag.find_words_recursive(next_node, rtn, new_letters, rack, next_hook)
        else:
            for key, child in node.children.items():
                if key in rack or key in {Gaddag.Node.EOW, Gaddag.Node.BREAK}:
                    new_rack = rack[:]
                    if key in new_rack:
                        new_rack.remove(key)
                    Gaddag.find_words_recursive(child, rtn, new_letters, new_rack, hook)

    # Converts the GADDAG nodes to dictionary
    def to_dict(self):
        def node_to_dict(node):
            node_dict = {
                'value': node.value,
                'children': {k: node_to_dict(v) for k, v in node.children.items()}
            }
            return node_dict

        return node_to_dict(self.root_node)

    # Recursively converts dictionary data back to Node objects
    @staticmethod
    def dict_to_node(node_dict):
        node = Gaddag.Node(node_dict['value'])
        for key, child_dict in node_dict['children'].items():
            node.children[int(key)] = Gaddag.dict_to_node(child_dict)
        return node 

    # Saves a GADDAG as a JSON file
    def save_as_json(self, filepath):
        gaddag_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(gaddag_dict, f, indent=4)
    
    # Loads the Gaddag from a JSON file
    @classmethod
    def load_from_json(cls, filepath):
        with open(filepath, 'r') as f:
            gaddag_dict = json.load(f)

        gaddag = cls()
        gaddag.root_node = cls.dict_to_node(gaddag_dict)
        return gaddag

    
    
