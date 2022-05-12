from coderone.dungeon.main import main

def train():
    game_stats, run_stats = main(['trainee_smith', 'coachPangolin', '--headless','--run_n_step=300000'])
    print(game_stats)
    print(run_stats)

if __name__ == '__main__':
    train()