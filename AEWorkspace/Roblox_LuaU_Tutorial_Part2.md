# Roblox LuaU Tutorial

## Advanced Concepts

### Advanced Object-Oriented Programming
Advanced Object-Oriented Programming in LuaU involves creating interfaces, abstract classes, and classes. For example:

```lua
local Weapon = {}
Weapon.__index = Weapon

function Weapon.new(name, damage, fireRate)
    local self = setmetatable({}, Weapon)
    self.Name = name
    self.Damage = damage
    self.FireRate = fireRate
    self.LastFireTime = 0
    return self
end

function Weapon:CanFire()
    local currentTime = tick()
    local timeSinceLastFire = currentTime - self.LastFireTime
    return timeSinceLastFire >= (1 / self.FireRate)
end

function Weapon:Fire(player, target)
    if not self:CanFire() then
        return false
    end

    self.LastFireTime = tick()
    target:TakeDamage(self.Damage, player)
    return true
end

local Rifle = setmetatable({}, {__index = Weapon})
Rifle.__index = Rifle

function Rifle.new(name, damage, fireRate, range)
    local self = Weapon.new(name, damage, fireRate)
    setmetatable(self, Rifle)
    self.Range = range
    self.BulletSpread = 0.1
    return self
end

function Rifle:Fire(player, target)
    local distance = (player.Position - target.Position).Magnitude
    if distance > self.Range then
        return false
    end

    return Weapon.Fire(self, player, target)
end

local assaultRifle = Rifle.new("AK-47", 35, 10, 300)
local canFire = assaultRifle:Fire(player, enemy)
```

### Module Scripts
ModuleScripts are used to organize professional Roblox code. They enable code reusability, maintainability, and team collaboration. For example:

```lua
local WeaponService = {}

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local activeWeapons = {}
local weaponConfigs = require(ReplicatedStorage.Configs.WeaponConfigs)

function WeaponService:Init()
    print("WeaponService initialized")
    self:SetupEvents()
end

local function validateWeaponData(player, weaponName)
    if not weaponConfigs[weaponName] then
        warn("Invalid weapon:", weaponName)
        return false
    end

    local inventory = player:FindFirstChild("Inventory")
    if not inventory or not inventory:FindFirstChild(weaponName) then
        warn("Player does not own weapon:", weaponName)
        return false
    end

    return true
end

function WeaponService:EquipWeapon(player, weaponName)
    if not validateWeaponData(player, weaponName) then
        return
    end

    if activeWeapons[player.UserId] then
        self:UnequipWeapon(player)
    end

    local weaponConfig = weaponConfigs[weaponName]
    local weaponModel = ReplicatedStorage.Weapons[weaponName]:Clone()

    weaponModel.Parent = player.Character
    activeWeapons[player.UserId] = {
        Name = weaponName,
        Model = weaponModel,
        Config = weaponConfig,
        Ammo = weaponConfig.MaxAmmo
    }

    print("Equipped", weaponName, "for", player.Name)
end

function WeaponService:FireWeapon(player, targetPosition)
    local weapon = activeWeapons[player.UserId]
    if not weapon then return end

    if weapon.Ammo <= 0 then
        return
    end

    local currentTime = tick()
    if weapon.LastFireTime and (currentTime - weapon.LastFireTime) < (1 / weapon.Config.FireRate) then
        return
    end

    weapon.LastFireTime = currentTime
    weapon.Ammo = weapon.Ammo - 1

    local character = player.Character
    local rootPart = character and character:FindFirstChild("HumanoidRootPart")
    if not rootPart then return end

    local rayOrigin = rootPart.Position
    local rayDirection = (targetPosition - rayOrigin).Unit * weapon.Config.Range

    local raycastParams = RaycastParams.new()
    raycastParams.FilterDescendantsInstances = {character}
    raycastParams.FilterType = Enum.RaycastFilterType.Blacklist

    local raycastResult = workspace:Raycast(rayOrigin, rayDirection, raycastParams)

    if raycastResult then
        local hitPart = raycastResult.Instance
        local hitCharacter = hitPart.Parent
        local humanoid = hitCharacter and hitCharacter:FindFirstChild("Humanoid")

        if humanoid then
            humanoid:TakeDamage(weapon.Config.Damage)
        end
    end

    local ReplicateEvent = ReplicatedStorage.Events.WeaponFired
    ReplicateEvent:FireAllClients(player, targetPosition, raycastResult)
end

function WeaponService:SetupEvents()
    local Events = ReplicatedStorage.Events

    Events.EquipWeapon.OnServerEvent:Connect(function(player, weaponName)
        self:EquipWeapon(player, weaponName)
    end)

    Events.FireWeapon.OnServerEvent:Connect(function(player, targetPosition)
        self:FireWeapon(player, targetPosition)
    end)
end

Players.PlayerRemoving:Connect(function(player)
    activeWeapons[player.UserId] = nil
end)

return WeaponService
```
