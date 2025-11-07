# StudentAI.py
# Minimal AI: Minimax + Alpha-Beta with simple heuristics
from random import randint
import copy
from Move import Move

class StudentAI:
    """
    Minimal AI: should beat/tie the provided Random AI ≥60%.
    Uses depth-limited minimax with alpha–beta pruning and simple move ordering.
    """

    def __init__(self, col=7, row=7, k=2, first=1, time=1200):
        # color: 1 = White (W) moves "up", 2 = Black (B) moves "down"
        self.col = col
        self.row = row
        self.k = k
        self.color = first
        self.opponent = {1: 2, 2: 1}
        self.board = None  # will be set by framework via get_move’s first call
        # search params (tweakable)
        self.max_depth = 3

    def get_move(self, move):
        """
        Called by the game loop with the opponent's last move.
        We must return our chosen Move.
        """
        # Initialize board lazily
        if self.board is None:
            # Board is created inside BoardClasses when GameLogic sets up the AIs.
            # Framework will assign self.board before our first decision by giving us
            # the opponent's move==[] for the very first call.
            # But to be safe, we fallback to importing BoardClasses and creating one.
            try:
                from BoardClasses import Board
                self.board = Board(self.col, self.row, self.k)
            except Exception:
                pass

        # If opponent moved, apply it locally
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])

        # Generate all our legal moves
        moves_grouped = self.board.get_all_possible_moves(self.color)
        if not moves_grouped:
            # No legal moves -> return an empty move (resigns)
            return Move([])

        # If only one move exists overall, take it quickly
        flat = [m for group in moves_grouped for m in group]
        if len(flat) == 1:
            chosen = flat[0]
            self.board.make_move(chosen, self.color)
            return chosen

        # Choose via minimax
        best_score = float("-inf")
        best_move = None

        # Order: captures first, then others
        candidate_moves = self._order_moves(flat)

        alpha = float("-inf")
        beta = float("inf")

        for m in candidate_moves:
            new_board = self._apply(self.board, m, self.color)
            score = self._min_value(new_board, self.max_depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, best_score)

        # Apply chosen move to our real board and return it
        self.board.make_move(best_move, self.color)
        return best_move

    # --- Minimax with alpha–beta ---

    def _max_value(self, board, depth, alpha, beta):
        winner = board.is_win(self.opponent[self.color])  # did opponent just lose?
        if winner != 0 or depth == 0:
            return self._evaluate(board)

        moves = board.get_all_possible_moves(self.color)
        if not moves:
            return self._evaluate(board)

        flat = [m for group in moves for m in group]
        flat = self._order_moves(flat)

        value = float("-inf")
        for m in flat:
            child = self._apply(board, m, self.color)
            value = max(value, self._min_value(child, depth - 1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_value(self, board, depth, alpha, beta):
        winner = board.is_win(self.color)  # did we just lose?
        if winner != 0 or depth == 0:
            return self._evaluate(board)

        moves = board.get_all_possible_moves(self.opponent[self.color])
        if not moves:
            return self._evaluate(board)

        flat = [m for group in moves for m in group]
        flat = self._order_moves(flat)  # ordering helps pruning even for opponent

        value = float("inf")
        for m in flat:
            child = self._apply(board, m, self.opponent[self.color])
            value = min(value, self._max_value(child, depth - 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    # --- Helpers ---

    def _apply(self, board, move, who):
        """Return a deep-copied board after applying `move` for player `who`."""
        nb = copy.deepcopy(board)
        nb.make_move(move, who)
        return nb

    def _order_moves(self, moves):
        """
        Prioritize capturing / longer sequences (likely captures),
        then potential promotions, then others.
        """
        def score_move(m: Move):
            seq = m.seq
            length = len(seq)
            # longer sequences likely include captures
            cap_bonus = max(0, length - 2)
            # promotion chance heuristic: reaching last row for our color
            promo = 0
            end_r, _ = seq[-1]
            if self.color == 1:  # White promotes at top (row 0)
                if end_r == 0:
                    promo = 1
            else:                # Black promotes at bottom (last row)
                if end_r == self.row - 1:
                    promo = 1
            return (cap_bonus * 10) + (promo * 5)

        return sorted(moves, key=score_move, reverse=True)

    # --- Evaluation ---

    def _evaluate(self, board):
        """
        Simple, fast, and effective for Minimal AI.
        Positive = good for self.color.
        """
        my_men = my_kings = opp_men = opp_kings = 0
        my_mob = opp_mob = 0
        my_center = opp_center = 0
        my_adv = opp_adv = 0

        # Count pieces and features
        for r in range(self.row):
            for c in range(self.col):
                piece = board.board[r][c]
                if piece == ".":
                    continue
                is_king = getattr(piece, "is_king", False)
                pcolor = getattr(piece, "color", ".")
                center = 1 if (1 <= r <= self.row - 2 and 1 <= c <= self.col - 2) else 0

                if pcolor == "W":
                    if is_king: 
                        w = "king"
                    if self.color == 1:
                        (my_kings if is_king else my_men).__class__
                    # both branches handled below
                # Count by perspective
                if pcolor == ("W" if self.color == 1 else "B"):
                    if is_king: my_kings += 1
                    else:       my_men   += 1
                    my_center += center
                    # advancement toward king row
                    if self.color == 1:
                        my_adv += (self.row - 1 - r)  # white moves up (smaller r), farther from bottom is "advanced"
                    else:
                        my_adv += r
                elif pcolor in ("W", "B"):
                    if is_king: opp_kings += 1
                    else:       opp_men   += 1
                    opp_center += center
                    if self.color == 1:
                        opp_adv += r
                    else:
                        opp_adv += (self.row - 1 - r)

        # Mobility (legal moves count)
        my_moves = board.get_all_possible_moves(self.color)
        opp_moves = board.get_all_possible_moves(self.opponent[self.color])
        my_mob = sum(len(g) for g in my_moves)
        opp_mob = sum(len(g) for g in opp_moves)

        # Terminal detection (board.is_win returns 1/2 winner or -1 for draw)
        term = board.is_win(self.color)
        if term == self.color:
            return 10_000
        elif term == self.opponent[self.color]:
            return -10_000
        elif term == -1:
            return 0

        # Weighted sum
        score  = 100 * (my_kings - opp_kings)
        score +=  50 * (my_men   - opp_men)
        score +=   5 * (my_mob   - opp_mob)
        score +=   2 * (my_center - opp_center)
        score +=   1 * (my_adv     - opp_adv)
        return score
