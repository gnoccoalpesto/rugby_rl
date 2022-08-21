# RUGBY-RL

## INTRO
rugby like simulation played using policy gradient method

ATTACKING AND DEFENDING TEAMS: 3 players
- DEF: plays randomly
- ATT: single actor(ball bearer) or whole team can learn

VICTORY CONDITIONS:
- time expires -> DEF
- ball passes try line -> ATT

FIELD:
- by default 13(long)* 19(wide)
- try line: 1/3 of the field (0,0 at upper left corner)

ACTIONS:
- advance (motion toward opponent endline)
- step back (toward its endline; also un-stuck the player)
- dodge (lateral left/right)
- pass the ball (finite maximum distance; also available back pass only rule)
- tackle (stops a opponent or steals the ball, if tackled has it)

POLICY
- policy gradient based (REINFORCE)
- softmax in actions preferences to describe policy mapping

NETWORK
[
keras.layers.Flatten(FIELD),
keras.layers.Dense(32, activation="softmax"),
keras.layers.Dense( activation='softmax',ACTIONS_NUMBER),
keras.layers.Reshape(ACTIONS{either single or multiple actors})
]

REWARDS
- game lost: -200,
- time step passed: -5,
- lost stolen by defense team: -5,
- game won: 200,
- advancing with the ball: +20,
- average game distance too big for ball pass: -40,
- ball stolen from defense team: 0,
- accomplishing ball pass: 0,
- keeping the position:-40

OPTIMIZER
- Adam
- using gradient ascent


## DEPENDENCIES
- field display: opencv (tested 4.6.0)
- loss dispay: matplotlib(tested 3.5.2), tkinter
NOTE: if unavailable, will only throw a warning
- training: tensorflow (tested 2.9.1)

## USAGE

{python3} scripts/decision.py
  [-h] help
  
  [-s] single actor: only ball bearer learns; default False
  
  [-w WIDTH] field width; default 9
  
  [-l LENGHT] field lenght; default 13
  
  [-d DURATION] game duration in steps; default 20
  
  [-g GAMES] number of games to play; default 500
  
  [-v] will print actions performed by players in terminal; default False
  
  [-t] will print field state in terminal; default False
  
  [-r] will display field image; default False
  
  [-f FRAME_DURATION] the lower, the higher the render frame rate; default 10ms
  
  [--RNG] plays a completely random match (no learning or loss printing)


## REPOSITORY

scripts/rugby_players.py: Player class definition, distane and average team distance functions

scripts/rugby_game.py: game definition, rules, actions, displaying

scripts/decision.py: launches the game, learning agent definition, argparser

