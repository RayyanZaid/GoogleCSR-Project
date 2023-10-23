from nba_api.stats.endpoints import playbyplay

def get_final_score(game_id):
    plays = playbyplay.PlayByPlay(game_id).get_normalized_dict()['PlayByPlay']
    home_score = 0
    visitor_score = 0

    for play in plays:
        if play['EVENTMSGTYPE'] == 13:  # Check for end of 4th quarter

            score_text = play['SCORE']
            if score_text:
                # Split the score string into home and visitor scores
                home, visitor = score_text.split('-')
                home_score = int(home.strip())
                visitor_score = int(visitor.strip())

    return home_score + visitor_score


game_id = '0021500492'
home_score, visitor_score = get_final_score(game_id)

print(f'Final Score: Home {home_score} - Visitor {visitor_score}')
