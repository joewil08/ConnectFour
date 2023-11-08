"""

"""

import sys
import random
import math
import numpy as np
import pygame


class Board:
    def __init__(self, num_of_columns, num_of_rows, num_to_connect):
        self.num_of_columns = num_of_columns
        self.num_of_rows = num_of_rows
        self.num_to_connect = num_to_connect
        self.board = np.zeros((num_of_rows, num_of_columns))

    def drop_piece(self, row, column, current_player):
        self.board[row][column] = current_player.piece

    def is_valid_move(self, column):
        if column in range(self.num_of_columns):
            return self.board[self.num_of_rows - 1][column] == 0
        return False

    def get_next_open_row(self, column):
        for row in range(self.num_of_rows):
            if self.board[row][column] == 0:
                return row

    def is_winning_move(self, row, column, current_player):
        # create a numpy array that contains the number of the current player's pieces that have to connect to win
        piece = current_player.piece
        winning_pattern = np.array([])
        for i in range(self.num_to_connect):
            winning_pattern = np.append(winning_pattern, piece)

        # check part of the row that the piece was dropped on
        for c in range(max(0, column - (self.num_to_connect - 1)),
                       min(column + self.num_to_connect, self.num_of_columns - (self.num_to_connect - 1))):
            if np.array_equal(self.board[row][c:c + self.num_to_connect], winning_pattern):
                return True

        # check part of the column that the piece was dropped on
        for r in range(max(0, row - (self.num_to_connect - 1)),
                       min(row + self.num_to_connect, self.num_of_rows - (self.num_to_connect - 1))):
            if np.array_equal(self.board[:, column][r:r + self.num_to_connect], winning_pattern):
                return True

        # check the / diagonal that the piece was dropped on
        positive_diagonal = np.diag(self.board, column - row)
        for d in range(len(positive_diagonal) - 3):
            if np.array_equal(positive_diagonal[d:d + self.num_to_connect], winning_pattern):
                return True

        # check the \ diagonal that the piece was dropped on
        negative_diagonal = np.diag(np.fliplr(self.board), abs(column - (self.num_of_columns - 1)) - row)
        for d in range(len(negative_diagonal) - (self.num_to_connect - 1)):
            if np.array_equal(negative_diagonal[d:d + self.num_to_connect], winning_pattern):
                return True

        return False

    def evaluate_window(self, window_current_piece_count, window_empty_count, window_opposing_piece_count):
        score = 0
        if window_current_piece_count == 4:
            score += 100
        elif window_current_piece_count == 3 and window_empty_count == 1:
            score += 5
        elif window_current_piece_count == 2 and window_empty_count == 2:
            score += 2

        if window_opposing_piece_count == 3 and window_empty_count == 1:
            score -= 4

        return score

    def score_position(self, column, row, current_player, opposing_player):
        score = 0

        # score center column
        center_array = self.board[:, self.num_of_columns // 2]
        center_count = np.count_nonzero(center_array == current_player.piece)
        score += center_count * 3

        # score horizontal
        for c in range(max(0, column - (self.num_to_connect - 1)),
                       min(column + self.num_to_connect, self.num_of_columns - (self.num_to_connect - 1))):
            window_current_piece_count = (
                np.count_nonzero(self.board[row][c:c + self.num_to_connect] == current_player.piece))
            window_empty_count = np.count_nonzero(self.board[row][c:c + self.num_to_connect] == 0)
            window_opposing_piece_count = np.count_nonzero(
                self.board[row][c:c + self.num_to_connect] == opposing_player.piece)
            score += self.evaluate_window(window_current_piece_count, window_empty_count, window_opposing_piece_count)

        # score vertical
        for r in range(max(0, row - (self.num_to_connect - 1)),
                       min(row + self.num_to_connect, self.num_of_rows - (self.num_to_connect - 1))):
            window_current_piece_count = (
                np.count_nonzero(self.board[:, column][r:r + self.num_to_connect] == current_player.piece))
            window_empty_count = np.count_nonzero(self.board[:, column][r:r + self.num_to_connect] == 0)
            window_opposing_piece_count = (
                np.count_nonzero(self.board[:, column][r:r + self.num_to_connect] == opposing_player.piece))
            score += self.evaluate_window(window_current_piece_count, window_empty_count, window_opposing_piece_count)

        # score / diagonal
        positive_diagonal = np.diag(self.board, column - row)
        for d in range(len(positive_diagonal) - (self.num_to_connect - 1)):
            window_current_piece_count = (
                np.count_nonzero(positive_diagonal[d:d + self.num_to_connect] == current_player.piece))
            window_empty_count = np.count_nonzero(positive_diagonal[d:d + self.num_to_connect] == 0)
            window_opposing_piece_count = (
                np.count_nonzero(positive_diagonal[d:d + self.num_to_connect] == opposing_player.piece))
            score += self.evaluate_window(window_current_piece_count, window_empty_count, window_opposing_piece_count)

        # score \ diagonal
        negative_diagonal = np.diag(np.fliplr(self.board), abs(column - (self.num_of_columns - 1)) - row)
        for d in range(len(negative_diagonal) - (self.num_to_connect - 1)):
            window_current_piece_count = (
                np.count_nonzero(negative_diagonal[d:d + self.num_to_connect] == current_player.piece))
            window_empty_count = np.count_nonzero(negative_diagonal[d:d + self.num_to_connect] == 0)
            window_opposing_piece_count = (
                np.count_nonzero(negative_diagonal[d:d + self.num_to_connect] == opposing_player.piece))
            score += self.evaluate_window(window_current_piece_count, window_empty_count, window_opposing_piece_count)

        return score

    def is_terminal_node(self, row, column, current_player, opposing_player):
        return (self.is_winning_move(row, column, current_player)
                or self.is_winning_move(row, column, opposing_player)
                or len(self.get_valid_locations()) == 0)

    def minimax(self, depth, alpha, beta, maximizing_player, row, column, current_player, opposing_player):
        valid_locations = self.get_valid_locations()
        is_terminal = self.is_terminal_node(row, column, current_player, opposing_player)
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.is_winning_move(row, column, current_player):
                    return None, 1000000000000000
                elif self.is_winning_move(row, column, opposing_player):
                    return None, -1000000000000000
                else:  # game over, no valid moves
                    return None, 0
            else:  # depth is 0
                return None, self.score_position(column, row, current_player, opposing_player)
        if maximizing_player:
            value = -math.inf
            col = random.choice(valid_locations)
            for c in valid_locations:
                r = self.get_next_open_row(c)
                b_copy = Board(self.num_of_columns, self.num_of_rows, self.num_to_connect)
                b_copy.board = self.board.copy()
                b_copy.drop_piece(r, c, current_player)
                new_score = b_copy.minimax(depth - 1, alpha, beta, False, r, c, current_player, opposing_player)[1]
                if new_score > value:
                    value = new_score
                    col = c
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return col, value
        else:  # minimizing player
            value = math.inf
            col = random.choice(valid_locations)
            for c in valid_locations:
                r = self.get_next_open_row(c)
                b_copy = Board(self.num_of_columns, self.num_of_rows, self.num_to_connect)
                b_copy.board = self.board.copy()
                b_copy.drop_piece(r, c, opposing_player)
                new_score = b_copy.minimax(depth - 1, alpha, beta, True, r, c, current_player, opposing_player)[1]
                if new_score < value:
                    value = new_score
                    col = c
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return col, value

    def get_valid_locations(self):
        valid_locations = []
        for c in range(self.num_of_columns):
            if self.is_valid_move(c):
                valid_locations.append(c)
        return valid_locations

    def pick_best_move(self, current_player, opposing_player):
        valid_locations = self.get_valid_locations()
        best_score = 0
        best_column = random.choice(valid_locations)
        for c in valid_locations:
            r = self.get_next_open_row(c)
            temp_board = Board(self.num_of_columns, self.num_of_rows, self.num_to_connect)
            temp_board.board = self.board.copy()
            print(temp_board.board)
            temp_board.board[r][c] = current_player.piece
            score = temp_board.score_position(c, r, current_player, opposing_player)
            if score > best_score:
                best_score = score
                best_column = c
        return best_column


class Player:
    def __init__(self, name, piece, is_a_computer):
        self.name = name
        self.piece = piece
        self.is_a_computer = is_a_computer


class Game:
    def __init__(self, board, player1, player2):
        self.board = board
        self.player1 = player1
        self.player2 = player2
        self.square_size = 70
        self.radius = int(self.square_size / 2 - 5)
        self.width = self.board.num_of_columns * self.square_size
        self.height = (self.board.num_of_rows + 1) * self.square_size
        self.size = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.size)
        self.blue = (47, 141, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.yellow = (255, 255, 0)
        self.green = (0, 255, 0)
        self.winner_font = pygame.font.SysFont("Arial Rounded MT Bold", 100)

    def draw_board(self):
        for c in range(self.board.num_of_columns):
            for r in range(self.board.num_of_rows):
                pygame.draw.rect(self.screen, self.blue,
                                 (c * self.square_size, r * self.square_size + self.square_size,
                                  self.square_size, self.square_size))
                pygame.draw.circle(self.screen, self.black,
                                   (int(c * self.square_size + self.square_size / 2),
                                    int(r * self.square_size + self.square_size + self.square_size / 2)),
                                   self.radius)
        for c in range(self.board.num_of_columns):
            for r in range(self.board.num_of_rows):
                if self.board.board[r][c] == 1:
                    pygame.draw.circle(self.screen, self.red,
                                       (int(c * self.square_size + self.square_size / 2),
                                        self.height - int(r * self.square_size + self.square_size / 2)),
                                       self.radius)
                elif self.board.board[r][c] == 2:
                    pygame.draw.circle(self.screen, self.yellow,
                                       (int(c * self.square_size + self.square_size / 2),
                                        self.height - int(r * self.square_size + self.square_size / 2)),
                                       self.radius)
        pygame.display.update()

    def print_board(self):
        # prints the board to the command line
        print(np.flipud(self.board.board))

    def process_response(self, column, current_player):
        # places the piece on the board and returns true if the player won the game with that move
        row = self.board.get_next_open_row(column)
        self.board.drop_piece(row, column, current_player)
        return self.board.is_winning_move(row, column, current_player)

    def change_player(self, current_player):
        # returns the player that is not the current player
        return self.player1 if current_player is self.player2 else self.player2

    def play_game(self):
        self.draw_board()
        pygame.display.update()
        current_player = random.choice([self.player1, self.player2]) # randomly select who gets first move
        opposing_player = self.player1 if current_player is self.player2 else self.player2
        game_over = False
        tie = False
        while not game_over:
            # Computer makes a move
            pygame.draw.rect(self.screen, self.black, (0, 0, self.width, self.square_size))
            if current_player.is_a_computer:
                column, minimax_score = self.board.minimax(4, -math.inf, math.inf, True, 0, 0, current_player,
                                                           opposing_player)
                if self.board.is_valid_move(column):
                    pygame.time.wait(500)
                    game_over = self.process_response(column, current_player)
                    if len(self.board.get_valid_locations()) == 0:  # true when all spaces are taken
                        tie = True
                        game_over = True
                    self.print_board()
                    self.draw_board()
                    if not game_over:
                        current_player = self.change_player(current_player)
                        opposing_player = self.change_player(opposing_player)

            # Human makes a move
            else:
                pygame.draw.rect(self.screen, self.black, (0, 0, self.width, self.square_size))
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()

                    if event.type == pygame.MOUSEMOTION:
                        pygame.draw.rect(self.screen, self.black, (0, 0, self.width, self.square_size))
                        pos_x = event.pos[0]
                        if current_player is self.player1:
                            pygame.draw.circle(self.screen, self.red, (pos_x, int(self.square_size / 2)), self.radius)
                        else:
                            pygame.draw.circle(self.screen, self.yellow, (pos_x, int(self.square_size / 2)),
                                               self.radius)
                        pygame.display.update()

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pos_x = event.pos[0]
                        column = int(math.floor(pos_x / self.square_size))
                        if self.board.is_valid_move(column):
                            game_over = self.process_response(column, current_player)
                            if len(self.board.get_valid_locations()) == 0:  # true when all spaces are taken
                                tie = True
                                game_over = True
                            self.print_board()
                            self.draw_board()
                            if not game_over:
                                current_player = self.change_player(current_player)
                                opposing_player = self.change_player(opposing_player)

        pygame.draw.rect(self.screen, self.black, (0, 0, self.width, self.square_size))

        if not tie:
            winner_color = self.red if current_player is self.player1 else self.yellow
            winner_label = self.winner_font.render(f"{current_player.name} wins!", 1, winner_color)
        else:
            winner_label = self.winner_font.render(f"Draw!", 1, self.green)
        winner_rect = winner_label.get_rect(center=(self.width / 2, 40))
        self.screen.blit(winner_label, winner_rect)
        pygame.display.update()
        pygame.time.wait(10000)
        return_to_menu = MainMenu()
        return_to_menu.run()


class MainMenu:
    def __init__(self):
        # screen
        self.width = 500
        self.height = 500
        self.size = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.size)
        # color and fonts
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.yellow = (255, 255, 0)
        self.blue = (0, 0, 255)
        self.green = (0, 255, 0)
        self.title_font = pygame.font.SysFont("Arial Rounded MT Bold", 120)
        self.option_font = pygame.font.SysFont("Verdana", 30)
        # number of columns buttons
        self.col_6_button = Button(self.white, 150, self.height / 4 + 100, 50, 40, "6")
        self.col_7_button = Button(self.green, 210, self.height / 4 + 100, 50, 40, "7")
        self.col_8_button = Button(self.white, 270, self.height / 4 + 100, 50, 40, "8")
        self.col_9_button = Button(self.white, 330, self.height / 4 + 100, 50, 40, "9")
        self.col_10_button = Button(self.white, 390, self.height / 4 + 100, 50, 40, "10")
        # number of rows buttons
        self.row_6_button = Button(self.green, 150, self.height / 4 + 150, 50, 40, "6")
        self.row_7_button = Button(self.white, 210, self.height / 4 + 150, 50, 40, "7")
        self.row_8_button = Button(self.white, 270, self.height / 4 + 150, 50, 40, "8")
        self.row_9_button = Button(self.white, 330, self.height / 4 + 150, 50, 40, "9")
        self.row_10_button = Button(self.white, 390, self.height / 4 + 150, 50, 40, "10")
        # number to win buttons
        self.win_3_button = Button(self.white, 150, self.height / 4 + 200, 50, 40, "3")
        self.win_4_button = Button(self.green, 210, self.height / 4 + 200, 50, 40, "4")
        self.win_5_button = Button(self.white, 270, self.height / 4 + 200, 50, 40, "5")
        self.win_6_button = Button(self.white, 330, self.height / 4 + 200, 50, 40, "6")
        # human or computer buttons
        self.p1_human_button = Button(self.green, 150, self.height / 4, 120, 40, "Human")
        self.p1_computer_button = Button(self.white, 280, self.height / 4, 120, 40, "AI")
        self.p2_human_button = Button(self.white, 150, self.height / 4 + 50, 120, 40, "Human")
        self.p2_computer_button = Button(self.green, 280, self.height / 4 + 50, 120, 40, "AI")
        # play button
        self.play_button = Button(self.green, self.width / 2 - 40, self.height - 90, 80, 50, "Play")
        # board and player settings
        self.num_of_columns = 7
        self.num_of_rows = 6
        self.num_to_connect = 4
        self.p1_is_a_computer = False
        self.p2_is_a_computer = True

    def draw_menu(self):
        # draw title
        title = self.title_font.render("Connect!", 1, (255, 255, 255))
        title_rect = title.get_rect(center=(self.width / 2, 60))
        self.screen.blit(title, title_rect)
        # draw red player options
        player1_label = self.option_font.render("Red:", 1, (255, 0, 0))
        self.screen.blit(player1_label, (30, self.height / 4))
        self.p1_human_button.draw(self.screen, True)
        self.p1_computer_button.draw(self.screen, True)
        # draw yellow player options
        player2_label = self.option_font.render("Yellow:", 1, (255, 255, 0))
        self.screen.blit(player2_label, (30, self.height / 4 + 50))
        self.p2_human_button.draw(self.screen, True)
        self.p2_computer_button.draw(self.screen, True)
        # draw column options
        columns_label = self.option_font.render("Cols:", 1, (0, 0, 255))
        self.screen.blit(columns_label, (30, self.height / 4 + 100))
        self.col_6_button.draw(self.screen, True)
        self.col_7_button.draw(self.screen, True)
        self.col_8_button.draw(self.screen, True)
        self.col_9_button.draw(self.screen, True)
        self.col_10_button.draw(self.screen, True)
        # draw row options
        rows_label = self.option_font.render("Rows:", 1, (0, 0, 255))
        self.screen.blit(rows_label, (30, self.height / 4 + 150))
        self.row_6_button.draw(self.screen, True)
        self.row_7_button.draw(self.screen, True)
        self.row_8_button.draw(self.screen, True)
        self.row_9_button.draw(self.screen, True)
        self.row_10_button.draw(self.screen, True)
        # draw num to win options
        connect_label = self.option_font.render("Win:", 1, (0, 0, 255))
        self.screen.blit(connect_label, (30, self.height / 4 + 200))
        self.win_3_button.draw(self.screen, True)
        self.win_4_button.draw(self.screen, True)
        self.win_5_button.draw(self.screen, True)
        self.win_6_button.draw(self.screen, True)
        # draw play button
        self.play_button.draw(self.screen, True)
        pygame.display.update()

    def run(self):
        running = True
        while running:
            self.draw_menu()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    # click play button
                    if self.play_button.is_over(event.pos):
                        b = Board(self.num_of_columns, self.num_of_rows, self.num_to_connect)
                        p1 = Player('Red', 1, self.p1_is_a_computer)
                        p2 = Player('Yellow', 2, self.p2_is_a_computer)
                        g = Game(b, p1, p2)
                        g.play_game()
                    # click player 1 human/computer buttons
                    elif self.p1_human_button.is_over(event.pos):
                        self.p1_is_a_computer = False
                        self.p1_human_button.set_color((0, 255, 0))
                        self.p1_computer_button.set_color(self.white)
                    elif self.p1_computer_button.is_over(event.pos):
                        self.p1_is_a_computer = True
                        self.p1_computer_button.set_color((0, 255, 0))
                        self.p1_human_button.set_color(self.white)
                    # click player 2 human/computer buttons
                    elif self.p2_human_button.is_over(event.pos):
                        self.p2_is_a_computer = False
                        self.p2_human_button.set_color((0, 255, 0))
                        self.p2_computer_button.set_color(self.white)
                    elif self.p2_computer_button.is_over(event.pos):
                        self.p2_is_a_computer = True
                        self.p2_computer_button.set_color((0, 255, 0))
                        self.p2_human_button.set_color(self.white)
                    # click columns buttons
                    elif self.col_6_button.is_over(event.pos):
                        self.num_of_columns = 6
                        self.col_6_button.set_color(self.green)
                        self.col_7_button.set_color(self.white)
                        self.col_8_button.set_color(self.white)
                        self.col_9_button.set_color(self.white)
                        self.col_10_button.set_color(self.white)
                    elif self.col_7_button.is_over(event.pos):
                        self.num_of_columns = 7
                        self.col_6_button.set_color(self.white)
                        self.col_7_button.set_color(self.green)
                        self.col_8_button.set_color(self.white)
                        self.col_9_button.set_color(self.white)
                        self.col_10_button.set_color(self.white)
                    elif self.col_8_button.is_over(event.pos):
                        self.num_of_columns = 8
                        self.col_6_button.set_color(self.white)
                        self.col_7_button.set_color(self.white)
                        self.col_8_button.set_color(self.green)
                        self.col_9_button.set_color(self.white)
                        self.col_10_button.set_color(self.white)
                    elif self.col_9_button.is_over(event.pos):
                        self.num_of_columns = 9
                        self.col_6_button.set_color(self.white)
                        self.col_7_button.set_color(self.white)
                        self.col_8_button.set_color(self.white)
                        self.col_9_button.set_color(self.green)
                        self.col_10_button.set_color(self.white)
                    elif self.col_10_button.is_over(event.pos):
                        self.num_of_columns = 10
                        self.col_6_button.set_color(self.white)
                        self.col_7_button.set_color(self.white)
                        self.col_8_button.set_color(self.white)
                        self.col_9_button.set_color(self.white)
                        self.col_10_button.set_color(self.green)
                    # click rows buttons
                    elif self.row_6_button.is_over(event.pos):
                        self.num_of_rows = 6
                        self.row_6_button.set_color(self.green)
                        self.row_7_button.set_color(self.white)
                        self.row_8_button.set_color(self.white)
                        self.row_9_button.set_color(self.white)
                        self.row_10_button.set_color(self.white)
                    elif self.row_7_button.is_over(event.pos):
                        self.num_of_rows = 7
                        self.row_6_button.set_color(self.white)
                        self.row_7_button.set_color(self.green)
                        self.row_8_button.set_color(self.white)
                        self.row_9_button.set_color(self.white)
                        self.row_10_button.set_color(self.white)
                    elif self.row_8_button.is_over(event.pos):
                        self.num_of_rows = 8
                        self.row_6_button.set_color(self.white)
                        self.row_7_button.set_color(self.white)
                        self.row_8_button.set_color(self.green)
                        self.row_9_button.set_color(self.white)
                        self.row_10_button.set_color(self.white)
                    elif self.row_9_button.is_over(event.pos):
                        self.num_of_rows = 9
                        self.row_6_button.set_color(self.white)
                        self.row_7_button.set_color(self.white)
                        self.row_8_button.set_color(self.white)
                        self.row_9_button.set_color(self.green)
                        self.row_10_button.set_color(self.white)
                    elif self.row_10_button.is_over(event.pos):
                        self.num_of_rows = 10
                        self.row_6_button.set_color(self.white)
                        self.row_7_button.set_color(self.white)
                        self.row_8_button.set_color(self.white)
                        self.row_9_button.set_color(self.white)
                        self.row_10_button.set_color(self.green)
                    # click num to win buttons
                    elif self.win_3_button.is_over(event.pos):
                        self.num_to_connect = 3
                        self.win_3_button.set_color(self.green)
                        self.win_4_button.set_color(self.white)
                        self.win_5_button.set_color(self.white)
                        self.win_6_button.set_color(self.white)
                    elif self.win_4_button.is_over(event.pos):
                        self.num_to_connect = 4
                        self.win_3_button.set_color(self.white)
                        self.win_4_button.set_color(self.green)
                        self.win_5_button.set_color(self.white)
                        self.win_6_button.set_color(self.white)
                    elif self.win_5_button.is_over(event.pos):
                        self.num_to_connect = 5
                        self.win_3_button.set_color(self.white)
                        self.win_4_button.set_color(self.white)
                        self.win_5_button.set_color(self.green)
                        self.win_6_button.set_color(self.white)
                    elif self.win_6_button.is_over(event.pos):
                        self.num_to_connect = 6
                        self.win_3_button.set_color(self.white)
                        self.win_4_button.set_color(self.white)
                        self.win_5_button.set_color(self.white)
                        self.win_6_button.set_color(self.green)


class Button:
    def __init__(self, color, x, y, width, height, text=''):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def draw(self, win, outline=None):
        # draw the button on the screen
        if outline:
            pygame.draw.rect(win, outline, (self.x - 2, self.y - 2, self.width + 4, self.height + 4), 0)

        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height), 0)

        if self.text != '':
            font = pygame.font.SysFont('Verdana', 30)
            text = font.render(self.text, 1, (0, 0, 0))
            win.blit(text, (
                self.x + (self.width / 2 - text.get_width() / 2), self.y + (self.height / 2 - text.get_height() / 2)))

    def is_over(self, pos):
        # returns true if the mouse position is over the button when called
        if self.x < pos[0] < self.x + self.width:
            if self.y < pos[1] < self.y + self.height:
                return True

        return False

    def set_color(self, color):
        self.color = color


if __name__ == "__main__":
    pygame.init()
    m = MainMenu()
    m.run()
