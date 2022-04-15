# Passive element type
#this has to be in a class namespace in order to use it in match/case szenarios -> this is annoying...very much so
class El:
    INDUCTOR: int = 1
    CAPACITOR: int = 2


#mode flags
class fcnmode:
    FIT: int = 1
    OUTPUT: int = 2
