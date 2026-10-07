# Roblox LuaU Tutorial

## Basic Concepts

### Introduction to LuaU
LuaU is the scripting language used in Roblox Studio. It is a fast, small, safe, gradually typed embeddable scripting language derived from Lua 5.1. It is designed to be easy to learn and use, making it an excellent choice for beginners in game development.

### Variables and Data Types

#### Variables
Variables are containers for storing data values. In LuaU, you can create variables using the `local` keyword. For example:

```lua
local playerName = "Maya"
local health = 100
local isAlive = true
local position = Vector3.new(0, 5, 0)
local red = Color3.fromRGB(255, 0, 0)
```

#### Data Types
LuaU includes the following data types:

- **Nil**: Represents non-existence or nothingness.
- **Booleans**: Have a value of either false or true.
- **Numbers**: Represent double-precision (64-bit) floating-point numbers.
- **Strings**: Are sequences of characters.
- **Tables**: Are arrays or dictionaries of any value except nil.
- **Enums**: Are fixed lists of items.

### Functions
Functions are reusable blocks of code that can be called by name. They can take arguments and return a result. For example:

```lua
local function heal(currentHealth, amount)
    local newHealth = currentHealth + amount
    if newHealth > 100 then
        newHealth = 100
    end
    return newHealth
end

print(heal(80, 30)) -- 100
```

### Properties and the Instance Model
Every object in the Explorer is an Instance with Properties you can read and change from code. For example:

```lua
local part = script.Parent
part.Anchored = true
part.BrickColor = BrickColor.new("Bright green")
part.Transparency = 0.2

local block = Instance.new("Part")
block.Size = Vector3.new(4, 4, 4)
block.Position = Vector3.new(0, 10, 0)
block.Anchored = true
block.Parent = workspace
```

### Events and :Connect()
Games are reactive — code should run when something happens. Events are used to connect functions to specific occurrences. For example:

```lua
local part = script.Parent

local function onTouched(otherPart)
    print(otherPart.Name .. " touched the part!")
end

part.Touched:Connect(onTouched)
```

### Services and game:GetService
Roblox groups its engine features into services. The correct way to reach a service is `game:GetService("ServiceName")`. For example:

```lua
local Players = game:GetService("Players")

Players.PlayerAdded:Connect(function(player)
    print(player.Name .. " joined the game.")
end)
```

### task.wait() and Loops
To pause a script, use `task.wait(seconds)`. When you write an infinite loop with `while true do`, you must include a `task.wait()` inside it. For example:

```lua
local part = script.Parent

while true do
    part.BrickColor = BrickColor.new("Bright red")
    task.wait(0.5)
    part.BrickColor = BrickColor.new("Bright blue")
    task.wait(0.5)
end
```
