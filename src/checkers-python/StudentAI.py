import copy
import math
import time
from random import choice
from Move import Move


class MCTSNode:
    """
    Node for Monte Carlo Tree Search.
    Holds a board state, whose turn it is next, and statistics.
    """

    def __init__(self, board, player_to_move, ai, parent=None, move=None):
        self.board = board
        self.player_to_move = player_to_move  
        self.ai = ai

        self.parent = parent
        self.move = move  

        self.children = []
        self.untried_moves = ai._get_all_moves(board, player_to_move)

        self.wins = 0.0
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        w_self = self.board.is_win(self.ai.color)
        w_opp = self.board.is_win(self.ai.opponent[self.ai.color])
        return (w_self != 0) or (w_opp != 0) or (len(self.untried_moves) == 0)

    def best_child_ucb(self, c):
        """Return child with highest UCB1 score."""
        best_score = float("-inf")
        best_child = None
        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.wins / child.visits
                explore = math.sqrt(math.log(self.visits) / child.visits)
                score = exploit + c * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


class StudentAI:
    """
    MCTS-based AI for Draft AI project.
    Uses time-bounded Monte Carlo Tree Search with UCB1.
    """

    def __init__(self, col=7, row=7, k=2, first=1, time=1200):
        self.col = col
        self.row = row
        self.k = k
        self.color = 2
        self.opponent = {1: 2, 2: 1}

        self.board = None  

        # MCTS parameters
        self.exploration_c = 1.4
        self.time_per_move = 0.7

    def get_move(self, move):
        """
        Called by the game loop with the opponent's last move.
        We must return our chosen Move.
        """

        if self.board is None:
            from BoardClasses import Board
            self.board = Board(self.col, self.row, self.k)
            self.board.initialize_game()

        if len(move) == 0:
            self.color = 1
        else:
            self.board.make_move(move, self.opponent[self.color])

        move_groups = self.board.get_all_possible_moves(self.color)
        if not move_groups:
            return Move([])

        flat_moves = [m for g in move_groups for m in g]
        if len(flat_moves) == 1:
            chosen = flat_moves[0]
            self.board.make_move(chosen, self.color)
            return chosen

        # --- MCTS SEARCH ---
        root_board = copy.deepcopy(self.board)
        root = MCTSNode(root_board, self.color, ai=self)

        end_time = time.time() + self.time_per_move

        while time.time() < end_time:
            node = root

            while node.is_fully_expanded() and node.children:
                node = node.best_child_ucb(self.exploration_c)

            if node.untried_moves and not node.is_terminal():
                m = node.untried_moves.pop()
                next_board = copy.deepcopy(node.board)
                next_board.make_move(m, node.player_to_move)
                next_player = self.opponent[node.player_to_move]
                child = MCTSNode(next_board, next_player, ai=self, parent=node, move=m)
                node.children.append(child)
                node = child

            winner = self._rollout(node.board, node.player_to_move)

            self._backpropagate(node, winner)

        if not root.children:
            chosen = flat_moves[0]
        else:
            best_child = max(root.children, key=lambda c: c.visits)
            chosen = best_child.move

        self.board.make_move(chosen, self.color)
        return chosen

    # ---------- helpers  ----------

    def _get_all_moves(self, board, player):
        groups = board.get_all_possible_moves(player)
        if not groups:
            return []
        return [m for g in groups for m in g]

    def _rollout(self, board, player_to_move, max_steps=80):
        """
        Play random moves until terminal or max_steps reached.
        Return the winner: self.color, self.opponent[self.color], or -1 for draw.
        """
        current_player = player_to_move
        steps = 0
        tmp_board = copy.deepcopy(board)

        while steps < max_steps:
            steps += 1

            w_self = tmp_board.is_win(self.color)
            w_opp = tmp_board.is_win(self.opponent[self.color])
            if w_self == self.color:
                return self.color
            if w_opp == self.opponent[self.color]:
                return self.opponent[self.color]
            if w_self == -1 or w_opp == -1:
                return -1  

            moves = self._get_all_moves(tmp_board, current_player)
            if not moves:
                return self.opponent[current_player]

            m = choice(moves)
            tmp_board.make_move(m, current_player)
            current_player = self.opponent[current_player]

        return self._material_winner(tmp_board)

    def _material_winner(self, board):
        """
        Heuristic winner if rollout doesn't finish in time:
        whoever has more material (kings weighted more).
        """
        my_color_char = "W" if self.color == 1 else "B"
        opp_color_char = "B" if self.color == 1 else "W"

        my_score = 0
        opp_score = 0

        for r in range(self.row):
            for c in range(self.col):
                piece = board.board[r][c]
                if piece == ".":
                    continue
                pcolor = getattr(piece, "color", ".")
                king = getattr(piece, "is_king", False)
                val = 3 if king else 1

                if pcolor == my_color_char:
                    my_score += val
                elif pcolor == opp_color_char:
                    opp_score += val

        if my_score > opp_score:
            return self.color
        elif opp_score > my_score:
            return self.opponent[self.color]
        else:
            return -1  

    def _backpropagate(self, node, winner):
        """
        Update visits and wins along the path back to the root.
        """
        while node is not None:
            node.visits += 1
            if winner == self.color:
                node.wins += 1.0
            elif winner == -1:
                node.wins += 0.5  
            node = node.parent
