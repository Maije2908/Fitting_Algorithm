# Drop-Down GUI_config
DROP_DOWN_WIDTH:        int = 10
DROP_DOWN_HEIGHT:       int = 1
DROP_DOWN_ELEMENTS =    ["Inductor", "Capacitor"]#, "Common-Mode Choke"]
DROP_DOWN_FONT =        "Helvetica 12 bold"

# Specification GUI_config
SPEC_WIDTH:             int = 100
SPEC_HEIGHT:            int = 2
SPEC_PADDING =          {'padx': 10, 'pady': 0}
HEADLINE_FONT =         "Helvetica 16 bold"

HEADLINE_PADDING =    {'padx': 10, 'pady': 10}
BUTTON_LEFT_PADDING =    {'padx': 10, 'pady': 10}
BUTTON_RIGHT_PADDING =    {'padx': 0, 'pady': 10}

# Entry GUI_config
ENTRY_WIDTH:            int = 50
ENTRY_HEIGHT:           int = 2
ENTRY_PADDING =         {'padx': 0, 'pady': 0}
ENTRY_FONT =            "Helvetica 12 bold"

# Colors
BCKGND_COLOR =          '#90c1ff'
WHITE =                 "#ffffff"

#log window width/height
LOG_HEIGHT  :           int = 20
LOG_WIDTH   :           int = 100

#Shunt or series through
SHUNT_THROUGH   :       int = 1
SERIES_THROUGH  :       int = 2

#Measurement System Z0
Z0              :       int = 50

#offset for filelist
FILELIST_ROW_OFFSET = 9




# Flags for the program to know which element is selected
class El:
    INDUCTOR: int = 1
    CAPACITOR: int = 2

