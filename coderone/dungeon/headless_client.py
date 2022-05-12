import time
import logging


logger = logging.getLogger(__name__)

class Client:
	def __init__(self, game, config):
		self.game = game
		self.config = config

		self.is_endless = self.config.get('endless', False)
		self.paused = False # self.config.get('start_paused', False)
		self.single_step = False # self.config.get('single_step', False)
		self.run_n_step = self.config.get('run_n_step', 0)

		self.total_step = 0
		self.game_played = 0
		self.game_won = [0, 0]
		self.cum_score = [0, 0]
		
	def _update(self):
		self.game.tick()
		self.total_step += 1

		stats = self.game.stats
		for p in stats.players.values():
			name = "{}{}".format(p.name, '(bot)' if p.is_bot else "")
			logger.info(f"{name} HP: {p.hp} / Ammo: {p.ammo} / Score: {p.score}, loc: ({p.position[0]}, {p.position[1]})")

		if self.game.is_over and (self.is_endless or self.total_step < self.run_n_step):
			self.game_played += 1
			if stats.winner_pid is not None:
				self.game_won[stats.winner_pid] += 1
			for i, p in enumerate(stats.players.values()):
				self.cum_score[i] = p.score
			self._reset_game()

	def run(self):
		try:
			while not self.game.is_over:
				if self.run_n_step > 0 and self.total_step >= self.run_n_step:
					break

				logger.info(f"game-step [{self.game.tick_counter}/{self.game.max_iterations}]")
				self._update()

		except KeyboardInterrupt:
			logger.info(f"user interrupted the game")
			pass

		return {'games_played':self.game_played, 'games_won':self.game_won}

	def _reset_game(self):
		self.game.generate_map()
