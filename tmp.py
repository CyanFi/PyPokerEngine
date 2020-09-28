def community_card_to_tuple(community_card):
    """
    :param community_card: round_state['community_card']
    :return: tuple of int (0..52)
    """
    for i in range(0, len(community_card)):
        community_card[i] = self.card_to_int(community_card[i])
    for i in range(0, 5 - len(community_card)):
        # if community card num <5, append 52 to fill out the rest
        community_card.append(52)
    return tuple(community_card)