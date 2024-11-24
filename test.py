from gaddag import Gaddag

gaddag = Gaddag()

# with open("SOWPODS.txt", "r") as file:
#     for line in file:
#         word = line.strip()
#         gaddag.add(word)

gaddag.add("camp")
gaddag.add("ape")
gaddag.add("ace")

rack = ["C", "A", "M", "P"]
left = ["A"]
right = ["E"]

res = gaddag.bridge_gap("camp", left, right, 1)
print(res)
# gaddag.save_as_json("gaddag_wordlist.txt")
