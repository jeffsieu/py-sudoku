from pickletools import float8
from random import shuffle, seed as random_seed, randrange
import sys
from typing import Iterable, List, Optional, Tuple, Union, cast


class UnsolvableSudoku(Exception):
    pass


class _SudokuSolver:
    def __init__(self, sudoku: 'Sudoku'):
        self.width = sudoku.width
        self.height = sudoku.height
        self.size = sudoku.size
        self.sudoku = sudoku

    def _solve(self) -> Optional['Sudoku']:
        blanks = self.__get_blanks()
        blank_count = len(blanks)
        are_blanks_filled = [False for _ in range(blank_count)]
        blank_fillers = self.__calculate_blank_cell_fillers(blanks)
        solution_board = self.__get_solution(
            Sudoku._copy_board(self.sudoku.board), blanks, blank_fillers, are_blanks_filled)
        solution_difficulty = 0
        if not solution_board:
            return None
        return Sudoku(self.width, self.height, board=solution_board, difficulty=solution_difficulty)

    def __calculate_blank_cell_fillers(self, blanks: List[Tuple[int, int]]) -> List[List[List[bool]]]:
        sudoku = self.sudoku
        valid_fillers = [[[True for _ in range(self.size)] for _ in range(
            self.size)] for _ in range(self.size)]
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
                    cell = sudoku.board[grid_row_start +
                                        y_offset][grid_col_start + x_offset]
                    if cell:
                        valid_fillers[row][col][cell - 1] = False
        return valid_fillers

    def __get_blanks(self) -> List[Tuple[int, int]]:
        blanks = []
        for i, row in enumerate(self.sudoku.board):
            for j, cell in enumerate(row):
                if cell == Sudoku._empty_cell_value:
                    blanks += [(i, j)]
        return blanks

    def __is_neighbor(self, blank1: Tuple[int, int], blank2: Tuple[int, int]) -> bool:
        row1, col1 = blank1
        row2, col2 = blank2
        if row1 == row2 or col1 == col2:
            return True
        grid_row1, grid_col1 = row1 // self.height, col1 // self.width
        grid_row2, grid_col2 = row2 // self.height, col2 // self.width
        return grid_row1 == grid_row2 and grid_col1 == grid_col2

    # Optimized version of above
    def __get_solution(self, board: List[List[Union[int, None]]], blanks: List[Tuple[int, int]], blank_fillers: List[List[List[bool]]], are_blanks_filled: List[bool]) -> Optional[List[List[int]]]:
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
            return cast(List[List[int]], board)

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
            solution_board = self.__get_solution(
                board, blanks, blank_fillers, are_blanks_filled)

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

    def __init__(self, width: int = 3, height: Optional[int] = None, board: Optional[Iterable[Iterable[Union[int, None]]]] = None, difficulty: Optional[float] = None, seed: int = randrange(sys.maxsize)):
        self.width = width
        self.height = height if height else width
        self.size = self.width * self.height
        self.__difficulty: float

        assert self.width > 0, 'Width cannot be less than 1'
        assert self.height > 0, 'Height cannot be less than 1'
        assert self.size > 1, 'Board size cannot be 1 x 1'

        if difficulty is not None:
            self.__difficulty = difficulty

        if board:
            blank_count = 0
            self.board: List[List[Union[int, None]]] = [
                [cell for cell in row] for row in board]
            for row in self.board:
                for i in range(len(row)):
                    if not row[i] in range(1, self.size + 1):
                        row[i] = Sudoku._empty_cell_value
                        blank_count += 1
            if difficulty == None:
                if self.validate():
                    self.__difficulty = blank_count / \
                        (self.size * self.size)
                else:
                    self.__difficulty = -2
        else:
            positions = list(range(self.size))
            random_seed(seed)
            shuffle(positions)
            self.board = [[(i + 1) if i == positions[j]
                           else Sudoku._empty_cell_value for i in range(self.size)] for j in range(self.size)]

    def solve(self, raising: bool = False) -> 'Sudoku':
        solution = _SudokuSolver(self)._solve() if self.validate() else None
        if solution:
            return solution
        elif raising:
            raise UnsolvableSudoku('No solution found')
        else:
            solution_board = Sudoku.empty(self.width, self.height).board
            solution_difficulty = -2
            return Sudoku(board=solution_board, difficulty=solution_difficulty)

    def validate(self) -> bool:
        row_numbers = [[False for _ in range(self.size)]
                       for _ in range(self.size)]
        col_numbers = [[False for _ in range(self.size)]
                       for _ in range(self.size)]
        box_numbers = [[False for _ in range(self.size)]
                       for _ in range(self.size)]

        for row in range(self.size):
            for col in range(self.size):
                cell = self.board[row][col]
                box = (row // self.height) * self.height + (col // self.width)
                if cell == Sudoku._empty_cell_value:
                    continue
                elif isinstance(cell, int):
                    if row_numbers[row][cell - 1]:
                        return False
                    elif col_numbers[col][cell - 1]:
                        return False
                    elif box_numbers[box][cell - 1]:
                        return False
                    row_numbers[row][cell - 1] = True
                    col_numbers[col][cell - 1] = True
                    box_numbers[box][cell - 1] = True
        return True

    @ staticmethod
    def _copy_board(board: Iterable[Iterable[Union[int, None]]]) -> List[List[Union[int, None]]]:
        return [[cell for cell in row] for row in board]

    @ staticmethod
    def empty(width: int, height: int):
        size = width * height
        board = [[Sudoku._empty_cell_value] * size] * size
        return Sudoku(width, height, board, 0)

    def difficulty(self, difficulty: float) -> 'Sudoku':
        assert 0 < difficulty < 1, 'Difficulty must be between 0 and 1'
        indices = list(range(self.size * self.size))
        shuffle(indices)
        problem_board = self.solve().board
        for index in indices[:int(difficulty * self.size * self.size)]:
            row_index = index // self.size
            col_index = index % self.size
            problem_board[row_index][col_index] = Sudoku._empty_cell_value
        return Sudoku(self.width, self.height, problem_board, difficulty)

    def show(self) -> None:
        if self.__difficulty == -2:
            print('Puzzle has no solution')
        if self.__difficulty == -1:
            print('Invalid puzzle. Please solve the puzzle (puzzle.solve()), or set a difficulty (puzzle.difficulty())')
        if not self.board:
            print('No solution')
        print(self.__format_board_ascii())

    def show_full(self) -> None:
        print(self.__str__())

    def __format_board_ascii(self) -> str:
        table = ''
        cell_length = len(str(self.size))
        format_int = '{0:0' + str(cell_length) + 'd}'
        for i, row in enumerate(self.board):
            if i == 0:
                table += ('+-' + '-' * (cell_length + 1) *
                          self.width) * self.height + '+' + '\n'
            table += (('| ' + '{} ' * self.width) * self.height + '|').format(*[format_int.format(
                x) if x != Sudoku._empty_cell_value else ' ' * cell_length for x in row]) + '\n'
            if i == self.size - 1 or i % self.height == self.height - 1:
                table += ('+-' + '-' * (cell_length + 1) *
                          self.width) * self.height + '+' + '\n'
        return table

    def __str__(self) -> str:
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


class DiagonalSudoku(Sudoku):
    def __init__(self, size: int = 3, board: Iterable[Iterable[int | None]] | None = None, difficulty: float | None = None, seed: int = randrange(sys.maxsize)):
        self.width = size
        self.height = size
        self.size = size * size
        self.__difficulty: float
        self.diagonal_left_to_right = [(i, i) for i in range(self.size)]
        self.diagonal_right_to_left = [(i, j) for i, j in enumerate(range(self.size-1,-1,-1))]

        assert self.width > 0, 'Width cannot be less than 1'
        assert self.height > 0, 'Height cannot be less than 1'
        assert self.size > 1, 'Board size cannot be 1 x 1'

        if difficulty is not None:
            self.__difficulty = difficulty

        if board:
            blank_count = 0
            self.board: List[List[Union[int, None]]] = [
                [cell for cell in row] for row in board]
            for row in self.board:
                for i in range(len(row)):
                    if row[i] not in range(1, self.size + 1):
                        row[i] = Sudoku._empty_cell_value
                        blank_count += 1
            for row, col in self.diagonal_left_to_right:
                if self.board[row][col] not in range(1, self.size+1):
                    self.board[row][col] = Sudoku._empty_cell_value
                    blank_count += 1
            for row, col in self.diagonal_right_to_left:
                if self.board[row][col] not in range(1, self.size+1):
                    self.board[row][col] = Sudoku._empty_cell_value
                    blank_count += 1
            if difficulty == None:
                if self.validate():
                    self.__difficulty = blank_count / \
                        (self.size * self.size)
                else:
                    self.__difficulty = -2
        else:
            positions = list(range(1, self.size+1))
            random_seed(seed)
            shuffle(positions)
            self.board = [[positions[j] if i == j else Sudoku._empty_cell_value for i in range(self.size)] for j in range(self.size)]

    def difficulty(self, difficulty: float) -> 'DiagonalSudoku':
        assert 0 < difficulty < 1, 'Difficulty must be between 0 and 1'
        indices = list(range(self.size * self.size))
        shuffle(indices)
        problem_board = self.solve().board
        for index in indices[:int(difficulty * self.size * self.size)]:
            row_index = index // self.size
            col_index = index % self.size
            problem_board[row_index][col_index] = Sudoku._empty_cell_value
        return DiagonalSudoku(self.width, problem_board, difficulty)

    def validate(self) -> bool:
        row_numbers = [[False for _ in range(self.size)] for _ in range(self.size)]
        col_numbers = [[False for _ in range(self.size)] for _ in range(self.size)]
        box_numbers = [[False for _ in range(self.size)] for _ in range(self.size)]
        diagonal_numbers = [[False for _ in range(self.size)] for _ in range(2)]

        for row in range(self.size):
            for col in range(self.size):
                cell = self.board[row][col]
                box = (row // self.height) * self.height + (col // self.width)
                if cell == Sudoku._empty_cell_value:
                    continue
                elif isinstance(cell, int):
                    if row_numbers[row][cell - 1] or col_numbers[col][cell - 1] or box_numbers[box][cell - 1]:
                        return False
                    row_numbers[row][cell - 1] = True
                    col_numbers[col][cell - 1] = True
                    box_numbers[box][cell - 1] = True
        for i in self.diagonal_left_to_right:
            cell = self.board[i[0]][i[1]]
            if cell == Sudoku._empty_cell_value:
                continue
            elif isinstance(cell, int):
                if diagonal_numbers[0][cell - 1]:
                    return False
                diagonal_numbers[0][cell - 1] = True
        for i in self.diagonal_right_to_left:
            cell = self.board[i[0]][i[1]]
            if cell == Sudoku._empty_cell_value:
                continue
            elif isinstance(cell, int):
                if diagonal_numbers[1][cell - 1]:
                    return False
                diagonal_numbers[1][cell - 1] = True
        return True

    def solve(self, raising: bool = False) -> 'DiagonalSudoku':
        solution = _DiagonalSudokuSolver(self)._solve() if self.validate() else None
        if solution:
            return solution
        elif raising:
            raise UnsolvableSudoku('No solution found')
        else:
            solution_board = DiagonalSudoku.empty(self.width, self.height).board
            solution_difficulty = -2
            return DiagonalSudoku(board=solution_board, difficulty=solution_difficulty)
    
    def show(self) -> None:
        if self.__difficulty == -2:
            print('Puzzle has no solution')
        if self.__difficulty == -1:
            print('Invalid puzzle. Please solve the puzzle (puzzle.solve()), or set a difficulty (puzzle.difficulty())')
        if not self.board:
            print('No solution')
        print(self.__format_board_ascii())

    def show_full(self) -> None:
        print(self.__str__())

    def __format_board_ascii(self) -> str:
        table = ''
        cell_length = len(str(self.size))
        row_square = []
        format_int = '{0:0' + str(cell_length) + 'd}'

        for i, row in enumerate(self.board):
            if i == 0:
                table += ('+-' + '-' * (cell_length + 1) * self.width) * self.height + '+' + '\n'
            
            for x in range(len(row)):
                if x != Sudoku._empty_cell_value:
                    if i == x:
                        row_square.append("\033[1m\033[4m{}\033[0m".format(format_int.format(row[x])))
                    elif self.diagonal_right_to_left[i][1] == x:
                        row_square.append("\033[1m\033[4m{}\033[0m".format(format_int.format(row[x])))
                    else:
                        row_square.append(format_int.format(row[x]))
                else:
                    row_square.append(' ' * cell_length)
            table += (('| ' + '{} ' * self.width) * self.height + '|').format(*row_square)  + '\n'
            row_square = []

            if i == self.size - 1 or i % self.height == self.height - 1:
                table += ('+-' + '-' * (cell_length + 1) * self.width) * self.height + '+' + '\n'
        return table

    def __str__(self) -> str:
        if self.__difficulty == -2:
            difficulty_str = 'INVALID PUZZLE (GIVEN PUZZLE HAS NO SOLUTION)'
        elif self.__difficulty == -1:
            difficulty_str = 'INVALID PUZZLE'
        elif self.__difficulty == 0:
            difficulty_str = 'SOLVED'
        else:
            difficulty_str = '{:.2f}'.format(self.__difficulty)
        return '''
------------------------------------
{}x{} ({}x{}) DIAGONAL SUDOKU PUZZLE
Difficulty: {}
------------------------------------
{}
        '''.format(self.size, self.size, self.width, self.height, difficulty_str, self.__format_board_ascii())


class _DiagonalSudokuSolver(_SudokuSolver):
    def __init__(self, sudoku: DiagonalSudoku):
        super().__init__(sudoku)
        self.diagonal_left_to_right = [(i, i) for i in range(self.size)]
        self.diagonal_right_to_left = [(i, j) for i, j in enumerate(range(self.size-1,-1,-1))]

    def _solve(self) -> Optional['DiagonalSudoku']:
        blanks = self.__get_blanks()
        blank_count = len(blanks)
        are_blanks_filled = [False for _ in range(blank_count)]
        blank_fillers = self.__calculate_blank_cell_fillers(blanks)
        solution_board = self.__get_solution(
            DiagonalSudoku._copy_board(self.sudoku.board), blanks, blank_fillers, are_blanks_filled)
        solution_difficulty = 0
        if not solution_board:
            return None
        return DiagonalSudoku(self.width, board=solution_board, difficulty=solution_difficulty)

    def __get_blanks(self) -> List[Tuple[int, int]]:
        blanks = []
        for i, row in enumerate(self.sudoku.board):
            for j, cell in enumerate(row):
                if cell == Sudoku._empty_cell_value:
                    blanks += [(i, j)]
        return blanks

    def __is_neighbor(self, blank1: Tuple[int, int], blank2: Tuple[int, int]) -> bool:
        """
        The function checks whether the cells are neighbors.
        Checks whether they are in one row, in one column,
        in one square whose dimensions are `self.width` and in the same diagonal.
        """
        row1, col1 = blank1
        row2, col2 = blank2
        if row1 == row2 or col1 == col2:
            return True
        grid_row1, grid_col1 = row1 // self.height, col1 // self.width
        grid_row2, grid_col2 = row2 // self.height, col2 // self.width
        if grid_row1 == grid_row2 and grid_col1 == grid_col2:
            return True
        if blank1 in self.diagonal_left_to_right and blank2 in self.diagonal_left_to_right:
            return True
        return blank1 in self.diagonal_right_to_left and blank2 in self.diagonal_right_to_left

    def __calculate_blank_cell_fillers(self, blanks: List[Tuple[int, int]]) -> List[List[List[bool]]]:
        sudoku = self.sudoku
        valid_fillers = [[[True for _ in range(self.size)] for _ in range(
            self.size)] for _ in range(self.size)]
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
            
            if (row, col) in self.diagonal_left_to_right:
                for i in self.diagonal_left_to_right:
                    same_diagonal = sudoku.board[row][col]
                    if i == (row,col) or not same_diagonal:
                        continue
                    valid_fillers[row][col][same_diagonal - 1] = False
            elif (row, col) in self.diagonal_right_to_left:
                for i in self.diagonal_right_to_left:
                    same_diagonal = sudoku.board[i[0]][i[1]]
                    if i == (row,col) or not same_diagonal:
                        continue
                    valid_fillers[row][col][same_diagonal - 1] = False
        return valid_fillers
    
    def __get_solution(self, board: List[List[Union[int, None]]], blanks: List[Tuple[int, int]], blank_fillers: List[List[List[bool]]], are_blanks_filled: List[bool]) -> Optional[List[List[int]]]:
        min_filler_count = None
        chosen_blank = None
        for i, blank in enumerate(blanks):
            x, y = blank
            if are_blanks_filled[i]:
                continue
            valid_filler_count = sum(blank_fillers[x][y])
            if valid_filler_count == 0:
                return None
            if not min_filler_count or valid_filler_count < min_filler_count:
                min_filler_count = valid_filler_count
                chosen_blank = blank
                chosen_blank_index = i
        
        if not chosen_blank:
            return cast(List[List[int]], board)
        
        row, col = chosen_blank
        are_blanks_filled[chosen_blank_index] = True
        revert_list = [False for _ in range(len(blanks))]
        for number in range(self.size):
            if not blank_fillers[row][col][number]:
                continue
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
            solution_board = self.__get_solution(
                board, blanks, blank_fillers, are_blanks_filled)
            if solution_board:
                return solution_board
            for i, blank in enumerate(blanks):
                if revert_list[i]:
                    blank_row, blank_col = blank
                    blank_fillers[blank_row][blank_col][number] = True
        are_blanks_filled[chosen_blank_index] = False
        board[row][col] = Sudoku._empty_cell_value
        return None