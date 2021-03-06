from enum import Enum
import os
import random
import pkgutil

class AssetType(Enum):
	IMAGE = 'images'
	SOUND = 'sounds'

class AssetManager:

	# Images
	AMMUNITION_IMAGE = "ammo.png"
	TREASURE_CHEST_IMAGE = "chest.png"
	BOMB_IMAGE = "bomb_64px.png"
	
	SOFT_BLOCK_IMAGE = "crate.png"
	ORE_BLOCK_IMAGE = "ore_block.png"
	INDESTRUCTABLE_BLOCK_IMAGE = "metal_block.png"


	SKELETON_IMAGE = "skelet_run_anim_f1.png"
	FIRE_IMAGE = "coin_anim_f0.png"
	EXPLOSION_IMAGE = "explosion.png"

	# Sounds
	EXP_SOUND = "explosion.mp3"

	FLOOR_TILES = ["floor_1.png", "floor_2.png", "floor_3.png", "floor_4.png", "floor_5.png", "floor_6.png", "floor_7.png", "floor_8.png"]
	PLAYER_AVATARS = [
		"wizard_m_64px.png",
		"p1_knight_64px.png",
		"wizard_f_64px.png",
		"p2_knight_64px_flipped.png", 
		"p2_knight_64px.png",
		"p2_knight_orange_64px_flipped.png", 
		]

	# --------Redifinition----------------
	FLOOR_TILES = ["ground_1.png", "ground_2.png", "ground_3.png", "ground_4.png", "pavilion.png", "ground_2_90.png",
				   "ground_2_180.png", "ground_2_270.png"]
	PLAYER_AVATARS = [
		"monkey.png",
		"wild_boar.png",
		"wizard_f_64px.png",
		"wild_boar.png",
		"wild_boar.png",
		"p2_knight_orange_64px_flipped.png",
	]

	
	def __init__(self, asset_dir:str):
		self.asset_dir = asset_dir

	@property
	def explosion(self):
		return self.asset(self.EXPLOSION_IMAGE, AssetType.IMAGE)

	# -------Redifinition-------
	@property
	def grass_tile(self):
		tile = random.choice(self.FLOOR_TILES)
		tile = self.FLOOR_TILES[2]
		return self.asset(tile, AssetType.IMAGE)

	@property
	def roud_straight_tile(self):
		tile = self.FLOOR_TILES[0]
		return self.asset(tile, AssetType.IMAGE)

	@property
	def roud_horizontal_tile(self):
		tile = self.FLOOR_TILES[3]
		return self.asset(tile, AssetType.IMAGE)

	@property
	def roud_corner_tile(self):
		tile = self.FLOOR_TILES[1]
		return self.asset(tile, AssetType.IMAGE)

	@property
	def pavilion_tile(self):
		tile = self.FLOOR_TILES[4]
		return self.asset(tile, AssetType.IMAGE)

	@property
	def roud_corner90_tile(self):
		tile = self.FLOOR_TILES[5]
		return self.asset(tile, AssetType.IMAGE)

	@property
	def roud_corner180_tile(self):
		tile = self.FLOOR_TILES[6]
		return self.asset(tile, AssetType.IMAGE)

	@property
	def roud_corner270_tile(self):
		tile = self.FLOOR_TILES[7]
		return self.asset(tile, AssetType.IMAGE)

	# -------Redifinition-------

	@property
	def ammunition(self):
		return self.asset(self.AMMUNITION_IMAGE, AssetType.IMAGE)

	@property
	def treasure(self):
		return self.asset(self.TREASURE_CHEST_IMAGE, AssetType.IMAGE)

	@property
	def bomb(self):
		return self.asset(self.BOMB_IMAGE, AssetType.IMAGE)

	@property
	def indestructible_block(self):
		return self.asset(self.INDESTRUCTABLE_BLOCK_IMAGE, AssetType.IMAGE)

	@property
	def soft_block(self):
		return self.asset(self.SOFT_BLOCK_IMAGE, AssetType.IMAGE)

	@property
	def ore_block(self):
		return self.asset(self.ORE_BLOCK_IMAGE, AssetType.IMAGE)

	@property
	def skeleton(self):
		return self.asset(self.SKELETON_IMAGE, AssetType.IMAGE)

	@property
	def fire(self):
		return self.asset(self.FIRE_IMAGE, AssetType.IMAGE)

	@property
	def explosion_sound(self):
		return self.asset(self.EXP_SOUND, AssetType.SOUND)


	def player_avatar(self, pid:int):
		avatar = self.PLAYER_AVATARS[pid % len(self.PLAYER_AVATARS)]
		return self.asset(avatar, AssetType.IMAGE)


	def asset(self, name, assetType: AssetType):
		# data = pkgutil.get_data(__name__, "templates/temp_file")
		return os.path.join(self.asset_dir, assetType.value, name)
