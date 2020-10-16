(define (problem testproblem)
(:domain eat)
(:objects
room1 - room
room2 - room
room3 - room
room4 - room
room5 - room
storage1 - storage
storage2 - storage
cereal1 - cereal
cheese1 - cheese
bread1 - bread
ketchup1 - ketchup
rawsteak1 - steak
butter1 - butter
salt1 - salt
sauce1 - sauce
potatoes1 - potatoes
garnish1 - garnish
knife1 - useable
fork1 - useable
stovetop1 - useable
pot1 - useable
hand1 - hand
hand2 - hand
key1 - useable
key2 - useable
key3 - useable
)
(:init

(agentIn room1)

(storageIn storage1 room3)
(storageIn storage2 room4)

(isStore room5)

(isRawSteak rawsteak1)
(isKnife knife1)
(isFork fork1)
(isStoveTop stovetop1)
(isPot pot1)

(objectIn cereal1 room1)
(objectIn cheese1 room2)
(objectIn bread1 room3)
(objectInStorage ketchup1 storage1)
(objectInStorage rawsteak1 storage2)
(objectInStorage salt1 storage1)
(objectInStorage butter1 storage1)
(objectInStorage sauce1 storage1)
(objectIn potatoes1 room5)
(objectIn garnish1 room1)

(useableIn knife1 room3)
(useableIn fork1 room2)
(useableIn stovetop1 room1)
(useableIn pot1 room4)

(handfree hand1)
(handfree hand2)

(keyForStorage key1 storage1)
(keyForStorage key2 storage2)
(useablein key1 room3)
(useablein key2 room2)
(useablein key3 room1)

)
(:goal
(and
(finished)
)
)
)