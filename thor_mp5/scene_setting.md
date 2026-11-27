# Scene setting

Scan of 120 AI2-THOR floor plans: kitchens `FloorPlan1`–`FloorPlan30`, living rooms `FloorPlan201`–`FloorPlan230`, bedrooms `FloorPlan301`–`FloorPlan330`, bathrooms `FloorPlan401`–`FloorPlan430`. Each room type has 30 scenes. Object counts below are the number of object instances in that scene. A type is listed under a room only when it appears in at least 10 of that room’s 30 scenes. Receptacle pairs are the most frequent `(object, receptacle)` placements across all 120 scenes, down to a count of 19.

## Runtime

| Item | Value |
| --- | --- |
| GPU | NVIDIA L40S |
| Driver | 610.43.02 |

`/etc/vulkan/icd.d` is empty. `/usr/share/vulkan/icd.d` contains:

- `broadcom_icd.x86_64.json`
- `freedreno_icd.x86_64.json`
- `intel_hasvk_icd.x86_64.json`
- `intel_icd.x86_64.json`
- `lvp_icd.x86_64.json`
- `nouveau_icd.x86_64.json`
- `nvidia_icd.x86_64.json`
- `panfrost_icd.x86_64.json`
- `powervr_mesa_icd.x86_64.json`
- `radeon_icd.x86_64.json`

## How many objects a scene holds

| Room | Scenes | Instances | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kitchen | 30 | 2088 | 54 | 68.5 | 69.60 | 90 |
| Living room | 30 | 1271 | 25 | 38 | 42.37 | 72 |
| Bedroom | 30 | 1228 | 29 | 42 | 40.93 | 56 |
| Bathroom | 30 | 1085 | 30 | 35.5 | 36.17 | 47 |
| All | 120 | 5672 | 25 | — | 47.27 | 90 |

Kitchens hold the most instances. Bathrooms are the tightest and the most even. The sparsest scene is `FloorPlan202` (25). The fullest are `FloorPlan16` and `FloorPlan30` (90 each).

### Kitchens

| Scene | Objects | Scene | Objects | Scene | Objects |
| --- | ---: | --- | ---: | --- | ---: |
| FloorPlan1 | 77 | FloorPlan11 | 64 | FloorPlan21 | 61 |
| FloorPlan2 | 74 | FloorPlan12 | 67 | FloorPlan22 | 70 |
| FloorPlan3 | 59 | FloorPlan13 | 85 | FloorPlan23 | 63 |
| FloorPlan4 | 55 | FloorPlan14 | 54 | FloorPlan24 | 74 |
| FloorPlan5 | 79 | FloorPlan15 | 62 | FloorPlan25 | 61 |
| FloorPlan6 | 68 | FloorPlan16 | 90 | FloorPlan26 | 57 |
| FloorPlan7 | 78 | FloorPlan17 | 79 | FloorPlan27 | 63 |
| FloorPlan8 | 76 | FloorPlan18 | 75 | FloorPlan28 | 62 |
| FloorPlan9 | 85 | FloorPlan19 | 69 | FloorPlan29 | 55 |
| FloorPlan10 | 67 | FloorPlan20 | 69 | FloorPlan30 | 90 |

### Living rooms

| Scene | Objects | Scene | Objects | Scene | Objects |
| --- | ---: | --- | ---: | --- | ---: |
| FloorPlan201 | 47 | FloorPlan211 | 35 | FloorPlan221 | 34 |
| FloorPlan202 | 25 | FloorPlan212 | 35 | FloorPlan222 | 34 |
| FloorPlan203 | 46 | FloorPlan213 | 46 | FloorPlan223 | 38 |
| FloorPlan204 | 63 | FloorPlan214 | 38 | FloorPlan224 | 64 |
| FloorPlan205 | 36 | FloorPlan215 | 53 | FloorPlan225 | 38 |
| FloorPlan206 | 45 | FloorPlan216 | 37 | FloorPlan226 | 34 |
| FloorPlan207 | 37 | FloorPlan217 | 37 | FloorPlan227 | 72 |
| FloorPlan208 | 37 | FloorPlan218 | 41 | FloorPlan228 | 35 |
| FloorPlan209 | 37 | FloorPlan219 | 50 | FloorPlan229 | 43 |
| FloorPlan210 | 41 | FloorPlan220 | 41 | FloorPlan230 | 52 |

### Bedrooms

| Scene | Objects | Scene | Objects | Scene | Objects |
| --- | ---: | --- | ---: | --- | ---: |
| FloorPlan301 | 54 | FloorPlan311 | 38 | FloorPlan321 | 29 |
| FloorPlan302 | 36 | FloorPlan312 | 33 | FloorPlan322 | 43 |
| FloorPlan303 | 51 | FloorPlan313 | 43 | FloorPlan323 | 43 |
| FloorPlan304 | 31 | FloorPlan314 | 30 | FloorPlan324 | 47 |
| FloorPlan305 | 37 | FloorPlan315 | 41 | FloorPlan325 | 56 |
| FloorPlan306 | 29 | FloorPlan316 | 29 | FloorPlan326 | 55 |
| FloorPlan307 | 48 | FloorPlan317 | 42 | FloorPlan327 | 42 |
| FloorPlan308 | 42 | FloorPlan318 | 51 | FloorPlan328 | 32 |
| FloorPlan309 | 44 | FloorPlan319 | 48 | FloorPlan329 | 33 |
| FloorPlan310 | 33 | FloorPlan320 | 36 | FloorPlan330 | 52 |

### Bathrooms

| Scene | Objects | Scene | Objects | Scene | Objects |
| --- | ---: | --- | ---: | --- | ---: |
| FloorPlan401 | 37 | FloorPlan411 | 37 | FloorPlan421 | 38 |
| FloorPlan402 | 41 | FloorPlan412 | 35 | FloorPlan422 | 39 |
| FloorPlan403 | 42 | FloorPlan413 | 42 | FloorPlan423 | 47 |
| FloorPlan404 | 32 | FloorPlan414 | 40 | FloorPlan424 | 34 |
| FloorPlan405 | 33 | FloorPlan415 | 35 | FloorPlan425 | 36 |
| FloorPlan406 | 30 | FloorPlan416 | 33 | FloorPlan426 | 38 |
| FloorPlan407 | 38 | FloorPlan417 | 34 | FloorPlan427 | 39 |
| FloorPlan408 | 34 | FloorPlan418 | 31 | FloorPlan428 | 33 |
| FloorPlan409 | 34 | FloorPlan419 | 31 | FloorPlan429 | 31 |
| FloorPlan410 | 39 | FloorPlan420 | 31 | FloorPlan430 | 41 |

## Object types in at least 10 scenes

The number in parentheses is how many of the 30 scenes of that room contain the type. Types at 30 are in every scene of the room.

### Kitchen (45 types)

In all 30: DishSponge, Bread, Toaster, GarbageCan, Floor, SinkBasin, Fork, Mug, Potato, Sink, Spoon, Fridge, Lettuce, SaltShaker, Egg, Spatula, StoveKnob, Plate, SoapBottle, Knife, CounterTop, ButterKnife, Microwave, Cabinet, CoffeeMachine, Faucet, Pot, Apple, LightSwitch, Pan, PepperShaker, Tomato, Cup, Bowl, StoveBurner.

| Type | Scenes |
| --- | ---: |
| Drawer | 28 |
| Window | 26 |
| DiningTable | 16 |
| Kettle | 15 |
| Stool | 14 |
| Chair | 12 |
| WineBottle | 10 |
| PaperTowelRoll | 10 |
| HousePlant | 10 |
| Ladle | 10 |

### Living room (32 types)

In all 30: GarbageCan, Floor, Pillow, FloorLamp, Television, CreditCard, Box, RemoteControl, LightSwitch, Laptop, Sofa, HousePlant, KeyChain, Painting.

| Type | Scenes |
| --- | ---: |
| Statue | 29 |
| Window | 28 |
| SideTable | 27 |
| ArmChair | 27 |
| CoffeeTable | 24 |
| Drawer | 23 |
| Chair | 20 |
| Newspaper | 18 |
| Vase | 18 |
| Watch | 16 |
| Shelf | 16 |
| DiningTable | 14 |
| DeskLamp | 13 |
| WateringCan | 13 |
| TVStand | 12 |
| Curtains | 10 |
| Plate | 10 |
| Dresser | 10 |

### Bedroom (32 types)

In all 30: CD, GarbageCan, Floor, Pillow, Bed, Book, AlarmClock, DeskLamp, CreditCard, Mirror, Pencil, Pen, Window, LightSwitch, Laptop, CellPhone, KeyChain.

| Type | Scenes |
| --- | ---: |
| Drawer | 29 |
| Blinds | 27 |
| Desk | 24 |
| Shelf | 23 |
| SideTable | 23 |
| Chair | 22 |
| Mug | 16 |
| BaseballBat | 13 |
| Bowl | 13 |
| Painting | 13 |
| Box | 12 |
| BasketBall | 11 |
| Dresser | 11 |
| TeddyBear | 10 |
| TennisRacket | 10 |

### Bathroom (31 types)

In all 30: ToiletPaper, HandTowelHolder, GarbageCan, Floor, Plunger, Sink, SinkBasin, Towel, SprayBottle, Cloth, SoapBar, HandTowel, TowelHolder, Mirror, SoapBottle, ToiletPaperHanger, Toilet, Faucet, LightSwitch, Candle, ScrubBrush.

| Type | Scenes |
| --- | ---: |
| CounterTop | 25 |
| ShowerHead | 24 |
| Bathtub | 21 |
| Window | 19 |
| BathtubBasin | 19 |
| ShowerDoor | 17 |
| Cabinet | 15 |
| Drawer | 15 |
| ShowerCurtain | 13 |
| ShowerGlass | 13 |

Floor, GarbageCan, and LightSwitch are in every scene of all four rooms. A CounterTop is in every kitchen and in 25 bathrooms. A coffee table is a living-room object (24 scenes); a side table is common in living rooms (27) and bedrooms (23).

## State-capable types among common pickupable objects

| Type | Properties |
| --- | --- |
| Apple | sliceable |
| Book | openable |
| Bottle | breakable, canFillWithLiquid |
| Bowl | breakable, canFillWithLiquid, dirtyable |
| Box | openable |
| Bread | sliceable |
| Candle | toggleable |
| CellPhone | breakable, toggleable |
| Cloth | dirtyable |
| Cup | breakable, canFillWithLiquid, dirtyable |
| Egg | breakable, sliceable |
| Kettle | canFillWithLiquid, openable |
| Laptop | breakable, openable, toggleable |
| Lettuce | sliceable |
| Mug | breakable, canFillWithLiquid, dirtyable |
| Pan | dirtyable |
| Plate | breakable, dirtyable |
| Pot | canFillWithLiquid, dirtyable |
| Potato | cookable, sliceable |
| Statue | breakable |
| Tomato | sliceable |
| Vase | breakable |
| WateringCan | canFillWithLiquid |
| WineBottle | breakable, canFillWithLiquid |

Sliceable food in this list: Apple, Bread, Egg, Lettuce, Potato, Tomato. Potato is the only cookable type. Openable among these pickupables: Book, Box, Kettle, Laptop. Fillable: Bottle, Bowl, Cup, Kettle, Mug, Pot, WateringCan, WineBottle. Dirtyable: Bowl, Cloth, Cup, Mug, Pan, Plate, Pot. Toggleable: Candle, CellPhone, Laptop.

## Most frequent (object, receptacle) pairs

Counts are over the 120 scenes. The list stops at 19.

| Object | Relation | Receptacle | Scenes |
| --- | --- | --- | ---: |
| GarbageCan | on/in | Floor | 120 |
| SideTable | on/in | Floor | 57 |
| Chair | on/in | Floor | 54 |
| SoapBottle | on/in | CounterTop | 51 |
| Drawer | on/in | Floor | 44 |
| Shelf | on/in | Floor | 33 |
| ArmChair | on/in | Floor | 33 |
| Sofa | on/in | Floor | 31 |
| FloorLamp | on/in | Floor | 30 |
| Bed | on/in | Floor | 30 |
| Pillow | on/in | Bed | 30 |
| HandTowel | on/in | HandTowelHolder | 30 |
| Towel | on/in | TowelHolder | 30 |
| Toilet | on/in | Floor | 30 |
| DiningTable | on/in | Floor | 29 |
| ScrubBrush | on/in | Floor | 29 |
| Plunger | on/in | Floor | 29 |
| CoffeeMachine | on/in | CounterTop | 27 |
| SaltShaker | on/in | CounterTop | 27 |
| Fridge | on/in | Floor | 27 |
| Spatula | on/in | CounterTop | 27 |
| PepperShaker | on/in | CounterTop | 27 |
| Toaster | on/in | CounterTop | 27 |
| DeskLamp | on/in | SideTable | 25 |
| CoffeeTable | on/in | Floor | 25 |
| Desk | on/in | Floor | 25 |
| ButterKnife | on/in | CounterTop | 24 |
| Faucet | on/in | Sink | 24 |
| Pillow | on/in | Sofa | 24 |
| Box | on/in | Floor | 24 |
| Laptop | on/in | Desk | 23 |
| Dresser | on/in | Floor | 23 |
| Pencil | on/in | Desk | 22 |
| Pen | on/in | Desk | 22 |
| Bread | on/in | CounterTop | 21 |
| AlarmClock | on/in | SideTable | 21 |
| Cabinet | on/in | Floor | 20 |
| Faucet | on/in | CounterTop | 20 |
| CreditCard | on/in | Desk | 20 |
| Bowl | on/in | CounterTop | 19 |

A garbage can stands on the floor in every scene. Kitchen countertops hold the soap bottle, coffee machine, salt and pepper, spatula, toaster, butter knife, bread, and bowl. Bedroom desks hold the laptop, pencil, pen, and credit card; side tables hold the desk lamp and alarm clock. Towels sit on their holders in every bathroom that has them, which is all 30.
