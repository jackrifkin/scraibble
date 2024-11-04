from gaddag import Gaddag

gaddag = Gaddag()

with open("SOWPODS.txt", "r") as file:
    for line in file:
        word = line.strip()
        gaddag.add(word)

# gaddag.save_as_json("gaddag_wordlist.txt")
