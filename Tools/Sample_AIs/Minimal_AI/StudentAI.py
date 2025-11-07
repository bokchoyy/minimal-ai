from Move import Move
import copy

class StudentAI:
    """
    Simple minimax + alpha-beta AI.
    Should win/tie ≥60% vs Random on 7x7,k=2.
    """

    def __init__(self, col=7, row=7, k=2, first=1, time=1200):
        self.col = col
        self.row = row
        self.k = k
        self.color = first  # 1 = Black, 2 = White
        self.opponent = {1: 2, 2: 1}
        self.board = None
        self.max_depth = 2

    def get_move(self, move):
        # lazy init
        if self.board is None:
            from BoardClasses import Board
            self.board = Board(self.col, self.row, self.k)
            self.board.initialize_game()
            # decide color by turn order
            self.color = 1 if len(move) == 0 else 2

        # apply opponent move if not first
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])

        moves_grouped = self.board.get_all_possible_moves(self.color)
        if not moves_grouped:
            return Move([])

        # flatten all moves
        flat = [m for g in moves_grouped for m in g]
        if len(flat) == 1:
            chosen = flat[0]
            self.board.make_move(chosen, self.color)
            return chosen

        alpha, beta = float("-inf"), float("inf")
        best_score, best_move = float("-inf"), None

        for m in self._order_moves(flat):
            new_board = self._apply(self.board, m, self.color)
            score = self._min_value(new_board, self.max_depth - 1, alpha, beta)
            if score > best_score:
                best_score, best_move = score, m
            alpha = max(alpha, best_score)
            if best_score >= beta:
                break

        self.board.make_move(best_move, self.color)
        return best_move

    # ---------- minimax ----------
    def _max_value(self, board, depth, alpha, beta):
        term = board.is_win(self.color)
        if term != 0 or depth == 0:
            return self._evaluate(board)

        moves = board.get_all_possible_moves(self.color)
        if not moves:
            return self._evaluate(board)

        value = float("-inf")
        for m in self._order_moves([mm for g in moves for mm in g]):
            child = self._apply(board, m, self.color)
            value = max(value, self._min_value(child, depth - 1, alpha, beta))
            alpha = max(alpha, value)
            if value >= beta:
                break
        return value

    def _min_value(self, board, depth, alpha, beta):
        term = board.is_win(self.opponent[self.color])
        if term != 0 or depth == 0:
            return self._evaluate(board)

        moves = board.get_all_possible_moves(self.opponent[self.color])
        if not moves:
            return self._evaluate(board)

        value = float("inf")
        for m in self._order_moves([mm for g in moves for mm in g]):
            child = self._apply(board, m, self.opponent[self.color])
            value = min(value, self._max_value(child, depth - 1, alpha, beta))
            beta = min(beta, value)
            if value <= alpha:
                break
        return value

    # ---------- helpers ----------
    def _apply(self, board, move, who):
        nb = copy.deepcopy(board)
        nb.make_move(move, who)
        return nb

    def _order_moves(self, moves):
        # prioritize captures and promotions
        def key(m):
            seq = m.seq
            length = len(seq)
            cap_bonus = max(0, length - 2) * 10
            end_r, _ = seq[-1]
            promo = 1 if (self.color == 1 and end_r == self.row - 1) or (self.color == 2 and end_r == 0) else 0
            return cap_bonus + promo * 5
        return sorted(moves, key=key, reverse=True)

    # ---------- evaluation ----------
    def _evaluate(self, board):
        my_men = my_kings = opp_men = opp_kings = 0
        my_center = opp_center = 0
        my_adv = opp_adv = 0

        for r in range(self.row):
            for c in range(self.col):
                piece = board.board[r][c]
                if piece.color == '.':
                    continue
                me_color = 'B' if self.color == 1 else 'W'
                opp_color = 'W' if self.color == 1 else 'B'

                if piece.color == me_color:
                    if piece.is_king: my_kings += 1
                    else: my_men += 1
                    if 1 <= r < self.row - 1 and 1 <= c < self.col - 1: my_center += 1
                    my_adv += r if me_color == 'B' else (self.row - 1 - r)
                elif piece.color == opp_color:
                    if piece.is_king: opp_kings += 1
                    else: opp_men += 1
                    if 1 <= r < self.row - 1 and 1 <= c < self.col - 1: opp_center += 1
                    opp_adv += r if opp_color == 'B' else (self.row - 1 - r)

        my_moves = board.get_all_possible_moves(self.color)
        opp_moves = board.get_all_possible_moves(self.opponent[self.color])
        my_mob = sum(len(g) for g in my_moves)
        opp_mob = sum(len(g) for g in opp_moves)

        term = board.is_win(self.color)
        if term == self.color: return 10000
        if term == self.opponent[self.color]: return -10000
        if term == -1: return 0

        score = 120 * (my_kings - opp_kings)
        score += 50 * (my_men - opp_men)
        score += 6 * (my_mob - opp_mob)
        score += 2 * (my_center - opp_center)
        score += 1 * (my_adv - opp_adv)
        return score
