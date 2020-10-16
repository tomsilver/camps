(define (domain eat)
    (:requirements :strips :typing :action-costs)
    (:types 
        cereal grilledcheese steak cheese bread ketchup steak salt sauce butter lobster potatoes garnish - object
        useable storage room hand - other
    )
    (:predicates
        (taken ?o - object)
        (holding ?o - useable ?h  - hand)
        (handfree ?h - hand)
        (objectIn ?o - object ?r - room)
        (useableIn ?o - useable ?r - room)
        (storageIn ?s - storage ?r - room)
        (keyForStorage ?k - useable ?s - storage)
        (agentIn ?r - room)
        (objectInStorage ?o - object ?s - storage)
        (isStore ?r - room)
        (isRawSteak ?o - steak)
        (isCookedSteak ?o - steak)
        (isGrilledCheese ?o - cheese)
        (isPot ?pot - useable)
        (isKnife ?knife - useable)
        (isFork ?fork - useable)
        (isStoveTop ?o - useable)
        (finished)
        (finished0)
        (finished1)
        (finished2)
    )
    (:functions
        (total-cost) - number
    )

    (:action TAKE
        :parameters (?room - room ?obj - object ?hand - hand)
        :precondition (and
            (agentIn ?room)
            (objectIn ?obj ?room)
            (not (isStore ?room))
            (handfree ?hand)
        )
        :effect (and
            (taken ?obj)
            (not (objectIn ?obj ?room))
            (increase (total-cost) 1)
        )
    )

    (:action BUY
        :parameters (?room - room ?obj - object ?hand - hand)
        :precondition (and
            (agentIn ?room)
            (objectIn ?obj ?room)
            (isStore ?room)
            (handfree ?hand)
        )
        :effect (and
            (taken ?obj)
            (not (objectIn ?obj ?room))
            (increase (total-cost) 5)
        )
    )

    (:action TAKEOUT
        :parameters (?room - room ?storage - storage ?obj - object ?h1 - hand ?h2 - hand ?key - useable)
        :precondition (and
            (agentIn ?room)
            (objectInStorage ?obj ?storage)
            (storageIn ?storage ?room)
            (holding ?key ?h1)
            (keyForStorage ?key ?storage)
            (handfree ?h2)
        )
        :effect (and
            (not (objectInStorage ?obj ?storage))
            (objectIn ?obj ?room)
            (increase (total-cost) 1)
        )
    )

    (:action PICKUP
        :parameters (?room - room ?obj - useable ?h1 - hand)
        :precondition (and
            (agentIn ?room)
            (useableIn ?obj ?room)
            (handfree ?h1)
        )
        :effect (and
            (holding ?obj ?h1)
            (not (useableIn ?obj ?room))
            (not (handfree ?h1))
            (increase (total-cost) 1)
        )
    )

    (:action PUTDOWN
        :parameters (?room - room ?obj - useable ?h1 - hand)
        :precondition (and
            (agentIn ?room)
            (holding ?obj ?h1)
        )
        :effect (and
            (not (holding ?obj ?h1))
            (useableIn ?obj ?room)
            (handfree ?h1)
            (increase (total-cost) 1)
        )
    )

    (:action GOTO
        :parameters (?from - room ?to - room)
        :precondition (and
            (agentIn ?from)
        )
        :effect (and
            (not (agentIn ?from))
            (agentIn ?to)
            (increase (total-cost) 1)
        )
    )

    (:action EATCEREAL
        :parameters (?obj - cereal)
        :precondition (and
            (taken ?obj)
        )
        :effect (and
            (finished)
            (finished0)
            (increase (total-cost) <FINISHCOST0>)
        )
    )

    (:action PREPGRILLEDCHEESE
        :parameters (?cheese - cheese ?bread - bread ?ketchup - ketchup ?knife - useable ?stovetop - useable ?h1 - hand ?h2 - hand)
        :precondition (and
            (taken ?cheese)
            (taken ?bread)
            (taken ?ketchup)
            (isKnife ?knife)
            (holding ?knife ?h1)
            (isStoveTop ?stovetop)
            (holding ?stovetop ?h2)
        )
        :effect (and
            (isGrilledCheese ?cheese)
            (increase (total-cost) 1)
        )
    )

    (:action EATGRILLEDCHEESE
        :parameters (?obj - cheese ?fork - useable ?knife - useable ?h1 - hand ?h2 - hand)
        :precondition (and
            (isGrilledCheese ?obj)
            (taken ?obj)
            (isKnife ?knife)
            (holding ?knife ?h1)
            (isFork ?fork)
            (holding ?fork ?h2)
        )
        :effect (and
            (finished)
            (finished1)
            (increase (total-cost) <FINISHCOST1>)
        )
    )

    (:action PREPSTEAK
        :parameters (?rawsteak - steak ?salt - salt ?sauce - sauce ?potatoes - potatoes ?garnish - garnish ?knife - useable ?h1 - hand)
        :precondition (and
            (isRawSteak ?rawsteak)
            (taken ?rawsteak)
            (taken ?sauce)
            (taken ?salt)
            (taken ?potatoes)
            (taken ?garnish)
            (isKnife ?knife)
            (holding ?knife ?h1)
        )
        :effect (and
            (not (isRawSteak ?rawsteak))
            (isCookedSteak ?rawsteak)
            (increase (total-cost) 1)
        )
    )

    (:action EATSTEAK
        :parameters (?steak - steak ?knife - useable ?fork - useable ?h1 - hand ?h2 - hand ?room - room)
        :precondition (and
            (isStore ?room)
            (agentIn ?room)
            (isCookedSteak ?steak)
            (taken ?steak)
            (isKnife ?knife)
            (holding ?knife ?h1)
            (isFork ?fork)
            (holding ?fork ?h2)
        )
        :effect (and
            (finished)
            (finished2)
            (increase (total-cost) <FINISHCOST2>)
        )
    )
)
