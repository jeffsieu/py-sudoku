from random import shuffle, seed as random_seed, randrange
import sys


class UnsolvableSudoku(Exception):
    pass


class _SudokuSolver:
    def __init__(self, sudoku):
        self.width = sudoku.width
        self.height = sudoku.height
        self.size = sudoku.size
        self.sudoku = sudoku

    def _solve(self, raising=False):
        blanks = self.__get_blanks()
        blank_count = len(blanks)
        are_blanks_filled = [False for _ in range(blank_count)]
        blank_fillers = self.__calculate_blank_cell_fillers(blanks)
        solution_board = self.__get_solution(
            Sudoku._copy_board(self.sudoku.board), blanks, blank_fillers, are_blanks_filled)
        solution_difficulty = 0
        if not solution_board:
            if raising:
                raise UnsolvableSudoku
            solution_board = Sudoku.empty(self.width, self.height).board
            solution_difficulty = -2
        return Sudoku(self.width, self.height, board=solution_board, difficulty=solution_difficulty)

    def __calculate_blank_cell_fillers(self, blanks):
        sudoku = self.sudoku
        valid_fillers = [[[True for _ in range(self.size)] for _ in range(self.size)] for _ in range(self.size)]
        for row, col in blanks:
            for i in range(self.size):
                same_row = sudoku.board[row][i]
                same_col = sudoku.board[i][col]
                if same_row and i != col:
                    valid_fillers[row][col][same_row - 1] = False
                if same_col and i != row:
                    valid_fillers[row][col][same_col - 1] = False
            grid_row, grid_col = row // sudoku.height, col // sudoku.width
            grid_row_start = grid_row * sudoku.height
            grid_col_start = grid_col * sudoku.width
            for y_offset in range(sudoku.height):
                for x_offset in range(sudoku.width):
                    if grid_row_start + y_offset == row and grid_col_start + x_offset == col:
                        continue
                    cell = sudoku.board[grid_row_start + y_offset][grid_col_start + x_offset]
                    if cell:
                        valid_fillers[row][col][cell - 1] = False
        return valid_fillers

    def __get_blanks(self):
        blanks = []
        for i, row in enumerate(self.sudoku.board):
            for j, cell in enumerate(row):
                if cell == Sudoku._empty_cell_value:
                    blanks += [(i, j)]
        return blanks

    def __valid_fillers_for(self, board, pos):
        row, col = pos
        fillers = list(range(1, self.size + 1))

        # Check same row and column
        for i in range(self.size):
            if board[row][i]:
                fillers[board[row][i] - 1] = Sudoku._empty_cell_value
            if board[i][col]:
                fillers[board[i][col] - 1] = Sudoku._empty_cell_value

        # Check same grid
        grid_row, grid_col = row // self.height, col // self.width
        row_start = grid_row * self.height
        col_start = grid_col * self.width
        for y_offset in range(self.height):
            for x_offset in range(self.width):
                if board[row_start + y_offset][col_start + x_offset]:
                    fillers[board[row_start + y_offset]
                        [col_start + x_offset] - 1] = Sudoku._empty_cell_value
        return [f for f in fillers if f != Sudoku._empty_cell_value]

    def __is_neighbor(self, blank1, blank2):
        row1, col1 = blank1
        row2, col2 = blank2
        if row1 == row2 or col1 == col2:
            return True
        grid_row1, grid_col1 = row1 // self.height, col1 // self.width
        grid_row2, grid_col2 = row2 // self.height, col2 // self.width
        return grid_row1 == grid_row2 and grid_col1 == grid_col2

    # Optimized version of above
    def __get_solution(self, board, blanks, blank_fillers, are_blanks_filled):
        min_filler_count = None
        chosen_blank = None
        for i, blank in enumerate(blanks):
            x, y = blank
            if are_blanks_filled[i]:
                continue
            valid_filler_count = sum(blank_fillers[x][y])
            if valid_filler_count == 0:
                # Blank cannot be filled with any number, no solution
                return None
            if not min_filler_count or valid_filler_count < min_filler_count:
                min_filler_count = valid_filler_count
                chosen_blank = blank
                chosen_blank_index = i

        if not chosen_blank:
            # All blanks have been filled with valid values, return this board as the solution
            return board

        row, col = chosen_blank

        # Declare chosen blank as filled
        are_blanks_filled[chosen_blank_index] = True

        # Save list of neighbors affected by the filling of current cell
        revert_list = [False for _ in range(len(blanks))]

        for number in range(self.size):
            # Only try filling this cell with numbers its neighbors aren't already filled with
            if not blank_fillers[row][col][number]:
                continue

            # Test number in this cell, number + 1 is used because number is zero-indexed
            board[row][col] = number + 1

            for i, blank in enumerate(blanks):
                blank_row, blank_col = blank
                if blank == chosen_blank:
                    continue
                if self.__is_neighbor(blank, chosen_blank) and blank_fillers[blank_row][blank_col][number]:
                    blank_fillers[blank_row][blank_col][number] = False
                    revert_list[i] = True
                else:
                    revert_list[i] = False
            solution_board = self.__get_solution(board, blanks, blank_fillers, are_blanks_filled)

            if solution_board:
                return solution_board


            # No solution found by having tested number in this cell
            # So we reallow neighbor cells to have this number filled in them
            for i, blank in enumerate(blanks):
                if revert_list[i]:
                    blank_row, blank_col = blank
                    blank_fillers[blank_row][blank_col][number] = True

        # If this point is reached, there is no solution with the initial board state,
        # a mistake must have been made in earlier steps

        # Declare chosen cell as empty once again
        are_blanks_filled[chosen_blank_index] = False
        board[row][col] = Sudoku._empty_cell_value

        return None


class Sudoku:
    _empty_cell_value = None

    def __init__(self, width=3, height=None, board=None, difficulty=-1, seed=randrange(sys.maxsize)):
        self.board = board
        self.width = width
        self.height = height
        if not height:
            self.height = width
        self.size = self.width * self.height
        self.__difficulty = difficulty

        assert self.width > 0, 'Width cannot be less than 1'
        assert self.height > 0, 'Height cannot be less than 1'
        assert self.size > 1, 'Board size cannot be 1 x 1'

        if board:
            blank_count = 0
            for row in self.board:
                for i in range(len(row)):
                    if not row[i] in range(1, self.size + 1):
                        row[i] = Sudoku._empty_cell_value
                        blank_count += 1
            if difficulty == -1:
                self.__difficulty = blank_count / self.size / self.size
        else:
            positions = list(range(self.size))
            random_seed(seed)
            shuffle(positions)
            self.board = [[(i + 1) if i == positions[j] else Sudoku._empty_cell_value for i in range(self.size)] for j in range(self.size)]

    def solve(self, raising=False):
        return _SudokuSolver(self)._solve(raising)

    def validate(self):
        row_numbers = [[False for _ in range(self.size)] for _ in range(self.size)]
        col_numbers = [[False for _ in range(self.size)] for _ in range(self.size)]
        box_numbers = [[False for _ in range(self.size)] for _ in range(self.size)]

        for row in range(self.size):
            for col in range(self.size):
                cell = self.board[row][col]
                box = (row // self.height) * self.height + (col // self.width)
                if cell == Sudoku._empty_cell_value:
                    continue
                if row_numbers[row][cell - 1]:
                    return False
                if col_numbers[col][cell - 1]:
                    return False
                if box_numbers[box][cell - 1]:
                    return False
                row_numbers[row][cell - 1] = True
                col_numbers[col][cell - 1] = True
                box_numbers[box][cell - 1] = True
        return True

    @staticmethod
    def _copy_board(board):
        return [[cell for cell in row] for row in board]

    @staticmethod
    def empty(width, height):
        size = width * height
        board = [[Sudoku._empty_cell_value] * size] * size
        return Sudoku(width, height, board, 0)

    def difficulty(self, difficulty):
        assert 0 < difficulty < 1, 'Difficulty must be between 0 and 1'
        indices = list(range(self.size * self.size))
        shuffle(indices)
        problem_board = self.solve().board
        for index in indices[:int(difficulty * self.size * self.size)]:
            row_index = index // self.size
            col_index = index % self.size
            problem_board[row_index][col_index] = Sudoku._empty_cell_value
        return Sudoku(self.width, self.height, problem_board, difficulty)

    def show(self):
        if self.__difficulty == -2:
            print('Puzzle has no solution')
        if self.__difficulty == -1:
            print('Invalid puzzle. Please solve the puzzle (puzzle.solve()), or set a difficulty (puzzle.difficulty())')
            return self
        if not self.board:
            print('No solution')
            return self
        print(self.__format_board_ascii())

    def show_full(self):
        print(self.__str__())

    def __format_board_ascii(self):
        table = ''
        cell_length = len(str(self.size))
        format_int = '{0:0' + str(cell_length) + 'd}'
        for i, row in enumerate(self.board):
            if i == 0:
                table += ('+-' + '-' * (cell_length + 1) * self.width) * self.height + '+' + '\n'
            table += (('| ' + '{} ' * self.width) * self.height + '|').format(*[format_int.format(x) if x != Sudoku._empty_cell_value else ' ' * cell_length for x in row]) + '\n'
            if i == self.size - 1 or i % self.height == self.height - 1:
                table += ('+-' + '-' * (cell_length + 1) * self.width) * self.height + '+' + '\n'
        return table


    def __str__(self):
        if self.__difficulty == -2:
            difficulty_str = 'INVALID PUZZLE (GIVEN PUZZLE HAS NO SOLUTION)'
        elif self.__difficulty == -1:
            difficulty_str = 'INVALID PUZZLE'
        elif self.__difficulty == 0:
            difficulty_str = 'SOLVED'
        else:
            difficulty_str = '{:.2f}'.format(self.__difficulty)
        return '''
---------------------------
{}x{} ({}x{}) SUDOKU PUZZLE
Difficulty: {}
---------------------------
{}
        '''.format(self.size, self.size, self.width, self.height, difficulty_str, self.__format_board_ascii())
